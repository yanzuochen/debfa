import tvm.relay as r
import tvm.ir as ir
from tvm.ir.instrument import pass_instrument
from tvm.ir import IRModule
import numpy as np
from tvm.relay.qnn.op import dequantize
from typing import Tuple
import sys
import os

import utils

known_ops = {ir.op.Op.get(x): x for x in ir.op.Op.list_op_names()}

def simple_convish_weight(conv):
    return conv.args[1]

def conv_shape(conv):
    return f'{simple_convish_weight(conv).checked_type.shape}'

def is_conv(call):
    return get_op_name(call.op) in ['nn.conv2d', 'qnn.conv2d']

def is_dense(call):
    return get_op_name(call.op) in ['nn.dense', 'qnn.dense']

def is_simple_convish(expr):
    if not isinstance(expr, r.Call):
        return False
    return is_conv(expr) or is_dense(expr)

def is_convish(expr):
    """A convish can be a [q]{nn.conv2d, nn.dense} or a {TupleGetItem(0)+nn.batch_norm,
    nn.bias_add}-wrapped [q]{nn.conv2d, nn.dense}.
    It represents a small logical group in which the operators' parameters' shape
    change together."""
    if is_simple_convish(expr):
        return True
    if isinstance(expr, r.TupleGetItem) and \
        expr.index == 0 and isinstance(expr.tuple_value, r.Call) and \
        get_op_name(expr.tuple_value.op) == 'nn.batch_norm' and \
        is_simple_convish(expr.tuple_value.args[0]):
        return True
    if not isinstance(expr, r.Call):
        return False
    op_name = get_op_name(expr.op)
    if op_name == 'nn.bias_add' and \
        isinstance(expr.args[0], r.Call) and is_simple_convish(expr.args[0]):
        return True
    return False

def convish_components(convish):
    assert is_convish(convish)
    if is_simple_convish(convish):
        return [convish]
    if isinstance(convish, r.TupleGetItem):
        return [convish.tuple_value.args[0], convish.tuple_value, convish]
    op_name = get_op_name(convish.op)
    if op_name == 'nn.bias_add':
        return [convish.args[0], convish]
    raise NotImplementedError(f'{op_name} is not supported')

def is_dense_convish(convish):
    return is_dense(convish_components(convish)[0])

def is_shape_preserving(call):
    op_name = get_op_name(call.op)
    return not any(x in op_name for x in ['pool'])

def convish_weight(convish):
    return simple_convish_weight(convish_components(convish)[0])

def is_layerish(expr):
    if is_any_type(expr, r.Call, r.Var, r.Tuple, r.Constant):
        return True
    if isinstance(expr, r.TupleGetItem) and \
            expr.index == 0 and \
            is_any_type(expr.tuple_value, 'nn.batch_norm', 'nn.dropout'):
        return True
    return False

def layerish_components(layerish):
    assert is_layerish(layerish)
    if isinstance(layerish, r.TupleGetItem):
        return [layerish.tuple_value, layerish]
    return [layerish]

def layerish_core(layerish):
    return layerish_components(layerish)[0]

def layerish_parents(layerish, layerishs_only=False, pred=None):
    core = layerish_core(layerish)
    if isinstance(core, r.Call):
        return [x for x in core.args if
                (not layerishs_only or is_layerish(x)) and
                (not pred or pred(x))]
    if isinstance(core, r.Tuple):
        return [x for x in core.fields if
                (not layerishs_only or is_layerish(x)) and
                (not pred or pred(x))]
    return []

class ConvishVisitor(r.ExprVisitor):
    def __init__(self, post_order=False):
        super().__init__()
        self.handled = set()  # TODO: Check correctness
        self.post_order = post_order

    def visit_convish(self, convish):
        raise NotImplementedError()

    def visit_maybe_convish(self, convish, superf):
        if convish in self.handled or not is_convish(convish):
            return superf(convish)
        [self.handled.add(c) for c in convish_components(convish)]
        if self.post_order:
            ret = superf(convish)
            self.visit_convish(convish)
            return ret
        self.visit_convish(convish)
        return superf(convish)

    def visit_call(self, call):
        return self.visit_maybe_convish(call, super().visit_call)

    def visit_tuple_getitem(self, t):
        return self.visit_maybe_convish(t, super().visit_tuple_getitem)

class LayerishVisitor(r.ExprVisitor):
    def __init__(self, post_order=False):
        super().__init__()
        # A set to allow the passthrough traversal behaviour for the first
        # n-1 components in a complex layerish.
        self.passthru = set()
        self.post_order = post_order

    def visit_layerish(self, layerish):
        raise NotImplementedError()

    def visit_maybe_layerish(self, layerish, superf):
        if layerish in self.passthru:
            return superf(layerish)
        if not is_layerish(layerish):
            raise NotImplementedError(f'{desc_expr(layerish)} is not a layerish')
        [self.passthru.add(c) for c in layerish_components(layerish)[:-1]]
        if self.post_order:
            ret = superf(layerish)
            self.visit_layerish(layerish)
            return ret
        self.visit_layerish(layerish)
        return superf(layerish)

    def visit_call(self, call):
        return self.visit_maybe_layerish(call, super().visit_call)

    def visit_tuple_getitem(self, t):
        return self.visit_maybe_layerish(t, super().visit_tuple_getitem)

    def visit_var(self, var):
        return self.visit_maybe_layerish(var, super().visit_var)

    def visit_tuple(self, var):
        # Handcrafted Relay tuples
        return self.visit_maybe_layerish(var, super().visit_tuple)

    def visit_constant(self, const):
        return self.visit_maybe_layerish(const, super().visit_constant)

class LayerishChildrenFinder(LayerishVisitor):
    def __init__(self):
        super().__init__()
        self.children = {}

    def visit_layerish(self, layerish):
        core = layerish_core(layerish)
        for parent in layerish_parents(layerish, layerishs_only=True):
            if parent not in self.children:
                self.children[parent] = set()
            self.children[parent].add(layerish)
        if layerish not in self.children:  # For the first visited layerish
            self.children[layerish] = set()

    def get_children_map(self):
        return self.children

    def get_children(self, layerish):
        if layerish not in self.children:
            raise KeyError(f'{desc_expr(layerish)}')
        return self.children[layerish]

class BPVisitor(LayerishVisitor):
    """A visitor that ensures every node's children are visited before itself.
    Suitable for backpropagation."""

    def __init__(self, expr):
        super().__init__()
        self.expr = expr
        self.handled = set()
        self.cf = LayerishChildrenFinder()

        self.cf.visit(self.expr)
        # [print(f'{desc_expr(x)}') for x in self.cf.get_children_map().keys()]

    def visit(self, expr):
        # A hack to disable memo_map as we want to defer the visited check
        self.memo_map = {}
        return super().visit(expr)

    def visit_maybe_layerish(self, layerish, superf):
        if layerish in self.passthru:
            return superf(layerish)
        if layerish in self.handled:
            return  # We're not a mutator so just return nothing
        children = self.cf.get_children(layerish)
        if not children.issubset(self.handled):
            return  # We're not a mutator so just return nothing
        [self.passthru.add(c) for c in layerish_components(layerish)[:-1]]
        self.handled.add(layerish)
        self.visit_layerish(layerish)
        return superf(layerish)

    def run(self):
        self.handled.clear()
        return super().visit(self.expr)

class ConstArgsReplacer(r.ExprMutator):
    def __init__(self, scalars_only=True, use_var_placeholders=True):
        super().__init__()
        self.recover_mode = False
        self.scalars_only = scalars_only
        self.use_var_placeholders = use_var_placeholders
        self.next_const_id = 54321
        self.orig_consts = {}
        self.placeholder_vars = {}

    @staticmethod
    def key(x):
        if isinstance(x, r.Var):
            return x.name_hint
        if isinstance(x, r.Constant):
            # Assumption: all items in a const have the same value
            return x.data.numpy().item(0)
        raise NotImplementedError()

    def make_placeholder(self, shape, dtype):
        const_id = self.next_const_id
        self.next_const_id += 1
        if self.use_var_placeholders:
            ret = r.var(f'exconst_{const_id}', shape=shape, dtype=str(dtype))
            self.placeholder_vars[self.key(ret)] = ret
            return ret
        else:
            return r.const(np.ones(shape, dtype=dtype) * const_id)

    def visit(self, expr):
        if isinstance(expr, r.Constant):
            # Skip the cache
            return self.visit_constant(expr)
        return super().visit(expr)

    def visit_constant_replace(self, const):
        if self.scalars_only and len(const.data.shape) != 0:
            return const
        if not self.use_var_placeholders and const.data.dtype not in ['int32', 'float32']:
            utils.warn(f'ignoring {const.data.dtype} constant ({const.data.shape})')
            return const
        data = const.data.numpy()
        placeholder = self.make_placeholder(data.shape, data.dtype)
        self.orig_consts[self.key(placeholder)] = const
        # print(f'replacing {const.data} ({const.data.shape}) -> {self.key(placeholder)}')
        return placeholder

    def visit_var_recover(self, var):
        key = self.key(var)
        if not (key in self.orig_consts):
            return var
        # print(f'recovering {key} -> {self.orig_consts[key]}')
        ret = self.orig_consts[key]
        del self.orig_consts[key]
        del self.placeholder_vars[key]
        return ret

    def visit_constant_recover(self, const):
        data = const.data.numpy()
        key = self.key(const)
        if not (key in self.orig_consts and np.all(data == data.item(0))):
            return const
        # print(f'recovering {key} -> {self.orig_consts[key]}')
        ret = self.orig_consts[key]
        del self.orig_consts[key]
        return ret

    def visit_constant(self, const):
        if self.recover_mode:
            return self.visit_constant_recover(const)
        return self.visit_constant_replace(const)

    def visit_var(self, var):
        if self.recover_mode:
            return self.visit_var_recover(var)
        return var

    def run(self, expr, recover_mode=False):
        self.recover_mode = recover_mode
        ret = self.visit(expr)
        if recover_mode:
            assert len(self.orig_consts) == 0, f'{len(self.orig_consts)} constants not recovered'
        return ret

    def transform_mod(self, mod, recover_mode=False):
        fn = mod['main']
        new_body = self.run(fn.body, recover_mode=recover_mode)
        new_params = [x for x in fn.params]
        if self.use_var_placeholders:
            if not recover_mode:
                new_params += list(self.placeholder_vars.values())
            else:
                new_params = [x for x in fn.params if not x.name_hint.startswith('exconst_')]
        fn = r.Function(new_params, new_body, fn.ret_type, fn.type_params, fn.attrs)
        mod = r.transform.InferType()(IRModule.from_expr(fn))
        return mod

@r.transform.function_pass(opt_level=0)
class QNNPreLegalize:
    # TODO: Replace with vars instead of adding epsilon

    class PreLegalizer(r.ExprMutator):
        def visit_qnn_concat(self, call):
            """Ensures that no input scale/zero_point is equal to the output."""
            data = self.visit(call.args[0])
            input_scales: r.Tuple = self.visit(call.args[1])
            input_zps: r.Tuple = self.visit(call.args[2])
            output_scale = self.visit(call.args[3])
            output_zp = self.visit(call.args[4])
            if all(isinstance(x, r.Constant) for x in [*input_scales.fields, *input_zps.fields, output_scale, output_zp]):
                orig_output_scale, i = output_scale.data.numpy(), 1
                while any(ir.structural_equal(s, output_scale) and ir.structural_equal(zp, output_zp) for s, zp in zip(input_scales, input_zps)):
                    output_scale = r.const(orig_output_scale + i*1e-8)
                    i += 1
            return r.qnn.op.concatenate(data, input_scales, input_zps, output_scale, output_zp, call.attrs['axis'])

        def visit_qnn_conv2d(self, call):
            op = self.visit(call.op)
            conv_args = [self.visit(x) for x in call.args]
            if is_any_type(conv_args[0], 'qnn.quantize') and is_any_type(conv_args[0].args[0], r.Var):
                quant_node, quant_args = conv_args[0], conv_args[0].args
                input_node, quant_zp = quant_args[0], quant_args[2]
                if ir.structural_equal(quant_zp, r.const(0)):
                    quant_zp = r.const(2)  # 1 doesn't work it seems
                    quant_node = r.Call(
                        self.visit(quant_node.op),
                        [input_node, quant_args[1], quant_zp],
                        attrs=quant_node.attrs
                    )
                    conv_args[0], conv_args[2] = quant_node, quant_zp
            # return r.qnn.op.conv2d(*conv_args, **call.attrs)
            return r.Call(op, conv_args, call.attrs)

        def visit_call(self, call):
            if call.op.name == 'qnn.concatenate':
                return self.visit_qnn_concat(call)
            if call.op.name == 'qnn.conv2d':
                return self.visit_qnn_conv2d(call)
            return super().visit_call(call)

    def transform_function(self, func, mod, ctx):
        func = self.PreLegalizer().visit(func)
        return type_inferred(func, inplace=True)

@pass_instrument
class PassBlocker:
    """Blocks all passes after a given pass (inclusive)."""

    def __init__(self, stop_before: str):
        self.stop_before = stop_before
        self.blocked_passes = []
        self.negate = False
        self.target_seen = False

    def enter_pass_ctx(self):
        self.target_seen = False
        if not self.negate:
            self.blocked_passes = []

    def should_run(self, mod, info: ir.transform.PassInfo):
        if info.name == 'sequential':
            return True
        if info.name == self.stop_before or self.target_seen:
            self.target_seen = True
            if not self.negate:
                self.blocked_passes.append(info.name)
            return self.negate
        return not self.negate

@pass_instrument
class PassIRExporter:
    def __init__(self, fname_prefix, passes, output_dir='./debug/pass-irs'):
        self.fname_prefix = fname_prefix
        self.passes = passes
        self.output_dir = output_dir

    def run_after_pass(self, mod, info: ir.transform.PassInfo):
        if info.name not in self.passes:
            return
        outfile = os.path.join(self.output_dir, f'{self.fname_prefix}-{info.name}.log')
        if os.path.exists(outfile):
            print(f'Warning: Overwriting {outfile}')
        utils.ensure_dir_of(outfile)
        mod = type_inferred(mod)
        with open(outfile, 'w+')as f:
            f.write(str(mod))

def get_op_name(op):
    return known_ops.get(op)

def is_any_type(expr, *types):
    is_call = isinstance(expr, r.Call)
    for type_or_opname in types:
        if isinstance(type_or_opname, str):
            if is_call and get_op_name(expr.op) == type_or_opname:
                return type_or_opname
        elif isinstance(expr, type_or_opname):
            return type_or_opname
    return False

def type_inferred(expr, inplace=False):
    assert isinstance(expr, r.Expr), 'type_inferred called on non-Expr'
    try:
        if inplace:
            r.transform.InferTypeLocal(expr)
            return expr
        expr_mod = r.transform.InferType()(IRModule.from_expr(expr))
        return expr_mod['main'].body
    except:
        print(f'Failed to infer type for...', file=sys.stderr)
        print(expr, end='', file=sys.stderr)
        print('^~~~~ ...this expression', file=sys.stderr)
        raise

def get_type(expr):
    assert expr, "get_type called on None"
    try:
        return expr.checked_type
    except:
        return type_inferred(expr).checked_type

def get_shape(expr):
    return get_type(expr).concrete_shape

def get_dtype(expr):
    return get_type(expr).dtype

def desc_expr_type(expr):
    typ = get_type(expr)
    if isinstance(typ, r.TensorType):
        return f'{typ.concrete_shape}@{typ.dtype}'
    return str(typ)

def _desc_expr(expr):
    lhs = expr.__class__.__name__
    if isinstance(expr, r.Call):
        arg_hint = ''
        if len(expr.args) >= 2:
            arg1 = expr.args[1]
            if isinstance(arg1, r.Var):
                arg_hint = f'<{arg1.name_hint}>'
        input_type = ''
        if len(expr.args) >= 1:
            input_type = desc_expr_type(expr.args[0])
        lhs = f'{get_op_name(expr.op)}{arg_hint}({input_type})'
    elif isinstance(expr, r.TupleGetItem):
        lhs = f'{_desc_expr(expr.tuple_value)}[{expr.index}]'
    elif isinstance(expr, r.Var):
        lhs = f'%{expr.name_hint}'
    return lhs

def desc_expr(expr):
    delim = ' -> '
    if isinstance(expr, r.Var):
        delim = ': '
    elif isinstance(expr, r.Tuple):
        delim = ''
    lhs = _desc_expr(expr)
    typ = desc_expr_type(expr)
    return f'{lhs}{delim}{typ}'

def desc_exprs(exprs):
    return [desc_expr(x) for x in exprs]

def print_expr(expr):
    print(desc_expr(expr))

def print_exprs(exprs):
    for x in exprs:
        print_expr(x)

def unquant_expr(qexpr) -> Tuple[r.Expr, r.Expr]:
    """Takes a QNN expr and returns a non-quantised equivalent expr
    as well as the dequantised result of the given expr.
    We also use a hack to populate checked_type onto the new exprs."""
    qop_name = get_op_name(qexpr.op)
    if qop_name in {'qnn.add', 'qnn.subtract', 'qnn.mul'}:
        op = ir.Op.get(qop_name[4:])
        qlhs, qrhs, qlhs_scale, qlhs_zero_point, qrhs_scale, qrhs_zero_point, qoutput_scale, qoutput_zero_point = qexpr.args
        lhs_axis, rhs_axis = qexpr.attrs['lhs_axis'], qexpr.attrs['rhs_axis']

        lhs = dequantize(qlhs, qlhs_scale, qlhs_zero_point, axis=lhs_axis)
        rhs = dequantize(qrhs, qrhs_scale, qrhs_zero_point, axis=rhs_axis)
        unqexpr = r.Call(op, [lhs, rhs])
        deqout = dequantize(qexpr, qoutput_scale, qoutput_zero_point)

        [type_inferred(x, inplace=True) for x in [lhs, rhs, unqexpr, deqout]]
        return unqexpr, deqout
    elif qop_name in {'qnn.conv2d', 'qnn.dense'}:
        qdata, qweight, qdata_z, qweight_z, qdata_s, qweight_s = qexpr.args
        dequant_scale = qdata_s * qweight_s
        attrs = {k: qexpr.attrs[k] for k in qexpr.attrs.keys() if k not in {'out_dtype', 'units'}}

        data = dequantize(qdata, qdata_s, qdata_z)
        weight = dequantize(qweight, qweight_s, qweight_z)
        unqexpr = r.nn.__dict__[qop_name[4:]](data, weight, **attrs)
        deqout = dequantize(qexpr, dequant_scale, r.const(0))

        [type_inferred(x, inplace=True) for x in [data, weight, unqexpr, deqout]]
        return unqexpr, deqout
    elif qop_name == 'qnn.concatenate':
        qdata, qdata_scales, qdata_zero_points, output_scale, output_zero_point = qexpr.args
        axis = qexpr.attrs['axis']

        inputs = [dequantize(x, s, z) for x, s, z in zip(qdata, qdata_scales, qdata_zero_points)]
        unqexpr = r.concatenate(inputs, axis)
        deqout = dequantize(qexpr, output_scale, output_zero_point)

        [type_inferred(x, inplace=True) for x in [unqexpr, deqout, *unqexpr.args]]
        return unqexpr, deqout
    else:
        raise ValueError(f'Don\'t know how to unquant {desc_expr(qexpr)}')
