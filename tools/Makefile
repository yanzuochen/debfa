MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

%.pkl: FORCE
	$(MAKEFILE_DIR)/sweep2bits.py $@

%-bits.json: FORCE
	$(MAKEFILE_DIR)/bits2rewards.py $@

%-superbits.csv: FORCE
	$(MAKEFILE_DIR)/bits2superbits.py '' '' --parse-out-fname $@

FORCE:
