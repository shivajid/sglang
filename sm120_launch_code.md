 python3 -m sglang.launch_server \
              --model $MODEL_NAME \
              --quantization fp8 \
              --tensor-parallel-size 8 \
              --pipeline-parallel-size 2 \
              --attention-backend dsa \
              --dsa-prefill-backend tilelang \
              --dsa-decode-backend tilelang \
              --enable-flashinfer-allreduce-fusion \
              --dp-size 8 \
              --enable-dp-attention \
              --nnodes 2 \
              --node-rank $POD_INDEX \
              --port 8000 \
              --dist-init-addr sglang-master-pod:5000 \
              --trust-remote-code \
              --kv-cache-dtype bfloat16 \
              --disable-radix-cache \
              --reasoning-parser glm45 \
              --tool-call-parser glm47 \
              --mem-fraction-static 0.85 \
              --page-size 64 \
              --host 0.0.0.0 

=============================
Error:
2026-06-28 23:41:27 DP1 PP1 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 4266, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 420, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 863, in init_model_worker
    self.init_all_cuda_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 846, in init_all_cuda_graphs
    self.tp_worker.init_cuda_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 351, in init_cuda_graphs
    self.model_runner.init_cuda_graphs(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 914, in init_cuda_graphs
    self.eager_runner = EagerRunner(self)
                        ^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/eager_runner.py", line 136, in __init__
    self.warmup()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/base_runner.py", line 214, in warmup
    self._pre_initialize_flashinfer_allreduce_workspace()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/base_runner.py", line 247, in _pre_initialize_flashinfer_allreduce_workspace
    pre_initialize_workspaces(
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 837, in pre_initialize_workspaces
    ensure_workspace_initialized(
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 669, in ensure_workspace_initialized
    backend = resolve_flashinfer_allreduce_fusion_backend(server_args)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 84, in resolve_flashinfer_allreduce_fusion_backend
    return _resolve_backend(backend, is_multi_node)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 50, in _resolve_backend
    raise ValueError(
ValueError: FlashInfer allreduce fusion requires SM90 or SM10X NVIDIA GPUs.

[2026-06-28 23:41:27 DP0 PP1 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 4266, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 420, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 863, in init_model_worker
    self.init_all_cuda_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 846, in init_all_cuda_graphs
    self.tp_worker.init_cuda_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 351, in init_cuda_graphs
    self.model_runner.init_cuda_graphs(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 914, in init_cuda_graphs
    self.eager_runner = EagerRunner(self)
                        ^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/eager_runner.py", line 136, in __init__
    self.warmup()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/base_runner.py", line 214, in warmup
    self._pre_initialize_flashinfer_allreduce_workspace()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/base_runner.py", line 247, in _pre_initialize_flashinfer_allreduce_workspace
    pre_initialize_workspaces(
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 837, in pre_initialize_workspaces
    ensure_workspace_initialized(
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 669, in ensure_workspace_initialized
    backend = resolve_flashinfer_allreduce_fusion_backend(server_args)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 84, in resolve_flashinfer_allreduce_fusion_backend
    return _resolve_backend(backend, is_multi_node)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py", line 50, in _resolve_backend
    raise ValueError(
ValueError: FlashInfer allreduce fusion requires SM90 or SM10X NVIDIA GPUs.
