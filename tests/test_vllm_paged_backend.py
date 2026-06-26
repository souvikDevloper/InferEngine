from inferengine.model.vllm_paged import VLLMPagedBackend


def test_vllm_paged_backend_builds_managed_server_command(monkeypatch):
    monkeypatch.setenv("INFERENGINE_VLLM_PORT", "8123")
    monkeypatch.setenv("INFERENGINE_MAX_MODEL_LEN", "2048")
    monkeypatch.setenv("INFERENGINE_VLLM_GPU_MEMORY_UTILIZATION", "0.88")
    monkeypatch.setenv("INFERENGINE_TENSOR_PARALLEL_SIZE", "2")

    backend = VLLMPagedBackend("Qwen/Qwen2.5-7B-Instruct")
    command = backend._serve_command()

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen2.5-7B-Instruct"]
    assert command[command.index("--port") + 1] == "8123"
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.88"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
