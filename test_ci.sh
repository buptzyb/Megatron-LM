
pip install jet-client --upgrade --extra-index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hw-joc-pypi/simple
pip install PyGithub
sudo chmod 777 -R /usr/local/lib/python3.12/dist-packages/deep_ep/

# Example: For latest-dev checkpointing tests:
# bash /opt/megatron-lm/tests/unit_tests/run_ci_test.sh \
#     --tag latest \
#     --environment dev \
#     --bucket 'tests/unit_tests/**/*.py'

# uv run --no-sync python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/assets_dir/logs/1/ --tee 0:3 --redirects 3 -m coverage run --data-file=.coverage.unit_tests --source=megatron/core -m pytest -xvs --ignore=tests/unit_tests/transformer/moe/__init__.py --ignore=tests/unit_tests/transformer/moe/conftest.py --ignore=tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py --ignore=tests/unit_tests/transformer/moe/test_aux_loss.py --ignore=tests/unit_tests/transformer/moe/test_grouped_mlp.py --ignore=tests/unit_tests/transformer/moe/test_moe_layer.py --ignore=tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py --ignore=tests/unit_tests/transformer/moe/test_multihot_indices_converter.py --ignore=tests/unit_tests/transformer/moe/test_routers.py --ignore=tests/unit_tests/transformer/moe/test_sequential_mlp.py --ignore=tests/unit_tests/transformer/moe/test_shared_experts.py --ignore=tests/unit_tests/transformer/moe/test_token_dispatcher.py --ignore=tests/unit_tests/transformer/moe/test_upcycling.py -m 'not experimental and not flaky_in_dev' tests/unit_tests/transformer

# uv run --no-sync python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/assets_dir/logs/1/ --tee 0:3 --redirects 3 -m coverage run --data-file=.coverage.unit_tests --source=megatron/core -m pytest -xvs -m 'not experimental and not flaky_in_dev' tests/unit_tests/transformer/test_cuda_graphs.py

# python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/logs_cuda_graphs/1/ --tee 0:3 --redirects 3 -m pytest -xvs -m 'not experimental and not flaky_in_dev' tests/unit_tests/transformer/test_cuda_graphs.py
# python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/logs_fp8_param/1/ --tee 0:3 --redirects 3 -m pytest -xvs -m 'not experimental and not flaky_in_dev' tests/unit_tests/test_fp8_param.py
# python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/logs_random/1/ --tee 0:3 --redirects 3 -m pytest -xvs -m 'not experimental and not flaky_in_dev' tests/unit_tests/tensor_parallel/test_random.py

#tests/unit_tests/transformer/test_submodule_callables.py::TestTransformerLayerSubmoduleCallables::test_1f1b_overlap[True-True-alltoall]
# python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/logs_submodule_callables/1/ --tee 0:3 --redirects 3 -m pytest -xvs -m 'not experimental and not flaky_in_dev' tests/unit_tests/transformer/test_submodule_callables.py

# cp /usr/local/lib/python3.12/dist-packages/torch/distributed/distributed_c10d.py /opt/megatron-lm/
# sudo cp /opt/megatron-lm/distributed_c10d.py /usr/local/lib/python3.12/dist-packages/torch/distributed/

# cp /opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/graph.py /opt/megatron-lm/graph_opt.py
# cp /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/graph.py /opt/megatron-lm/graph_usr.py
# cp /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/distributed.py /opt/megatron-lm/distributed_usr.py
# cp /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py /opt/megatron-lm/dot_product_attention_usr.py
# cp /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/backends.py /opt/megatron-lm/backends_usr.py

# sudo cp /opt/megatron-lm/graph_opt.py /opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/graph.py 
# sudo cp /opt/megatron-lm/graph_usr.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/graph.py
# sudo cp /opt/megatron-lm/distributed_usr.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/distributed.py
# sudo cp /opt/megatron-lm/dot_product_attention_usr.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py
# sudo cp /opt/megatron-lm/backends_usr.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/backends.py

# cp /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/graph.py /home/scratch.robinz_sw/LLM/TransformerEngine/transformer_engine/pytorch/graph.py
# sudo cp /home/scratch.robinz_sw/LLM/TransformerEngine/transformer_engine/pytorch/graph.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/graph.py


# USE_MNNVL=0 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/logs_cuda_graphs/1/ --tee 0:3 --redirects 3 -m pytest -xvs -m 'not experimental and not flaky' tests/unit_tests/transformer/test_cuda_graphs.py
# python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --master_addr localhost --master_port 6000 --node_rank 0 --log-dir /opt/megatron-lm/logs_fp8_param/1/ --tee 0:3 --redirects 3 -m pytest -xvs -m 'not experimental and not flaky' tests/unit_tests/test_fp8_param.py


# python -m tests.test_utils.python_scripts.generate_local_jobs --environment dev --scope mr
CLUSTER=dgxh100_coreweave bash test_cases/moe/gpt3_moe_mcore_te_tp4_ep2_etp2_pp2_scoped_cudagraph.sh
