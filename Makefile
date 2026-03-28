ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: regression regression-cpu regression-gpu regression-gpu-quick gpu-sweep gpu-sweep-quick

regression: regression-cpu regression-gpu

regression-cpu:
	$(MAKE) -C $(ROOT_DIR)Mac/cpu regression

regression-gpu:
	$(MAKE) -C $(ROOT_DIR)Mac/gpu real-bundle-regression

regression-gpu-quick:
	$(MAKE) -C $(ROOT_DIR)Mac/gpu real-bundle-regression-quick

gpu-sweep:
	$(MAKE) -C $(ROOT_DIR)Mac/gpu integration-sweep

gpu-sweep-quick:
	$(MAKE) -C $(ROOT_DIR)Mac/gpu integration-sweep-quick