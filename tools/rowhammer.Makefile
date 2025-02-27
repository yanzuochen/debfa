MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

FORCE:

%.csv: FORCE
	$(MAKEFILE_DIR)/rh_attack.py --quiet --export $@
