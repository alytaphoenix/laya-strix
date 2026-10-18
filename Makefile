REMOTE ?= llm_server
REMOTE_DIR ?= laya-strix
PYTHON ?= .venv/bin/python
PORT ?= 8753
REFERENCE ?= http://127.0.0.1:8756
CANDIDATE ?= http://127.0.0.1:8754

.PHONY: help export simplify mask audit compile-remote serve bench parity train-cpu train-help dist clean

help:
	@echo "laya-strix targets"
	@echo "  export         static encoder/head ONNX from the English checkpoint"
	@echo "  simplify       onnxsim the encoder"
	@echo "  mask           replace the mask GatherND"
	@echo "  audit          print the encoder operator inventory"
	@echo "  compile-remote compile the BF16 EP context on $(REMOTE)"
	@echo "  serve          run the Jev service locally on :$(PORT)"
	@echo "  bench          latency probe against the local :$(PORT)"
	@echo "  parity         compare $(REFERENCE) and $(CANDIDATE) (tunnel)"
	@echo "  train-cpu      one-epoch frozen-encoder smoke run"
	@echo "  train-help     laya-npu train options"
	@echo "  dist           build the 0.1.0 wheel and sdist"

export:
	$(PYTHON) tools/export_split_static.py --model english \
		--out artifacts/onnx/english_split_256 --seq-len 256 --markers 8 --opset 17

simplify:
	$(PYTHON) tools/simplify_graph.py \
		--model artifacts/onnx/english_split_256/encoder.onnx \
		--out artifacts/onnx/english_split_256/encoder_sim.onnx \
		--verify artifacts/onnx/english_split_256/fixtures.npz

mask:
	$(PYTHON) tools/rewrite_mask.py \
		--model artifacts/onnx/english_split_256/encoder_sim.onnx \
		--out artifacts/onnx/english_split_256/encoder_sim_nomask.onnx \
		--verify artifacts/onnx/english_split_256/fixtures.npz

audit:
	$(PYTHON) tools/audit_graph.py artifacts/onnx/english_split_256/encoder_sim_nomask.onnx

compile-remote:
	ssh $(REMOTE) 'cd ~/$(REMOTE_DIR) && bash deploy/compile_final.sh'

serve:
	$(PYTHON) -m laya_strix.cli serve --host 127.0.0.1 --port $(PORT)

bench:
	$(PYTHON) tools/bench.py --url http://127.0.0.1:$(PORT) --requests 20

parity:
	$(PYTHON) tools/parity.py --reference $(REFERENCE) \
		--candidate $(CANDIDATE) \
		--probes ../forseti/crew/eval/laya-probe.jsonl

train-cpu:
	bash training/run_example_cpu.sh

train-help:
	bash deploy/train_laya.sh --help

dist:
	uv build
	@ls -lh dist/

clean:
	rm -rf dist build
