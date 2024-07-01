pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install peft
pip install datasets
pip install evaluate
pip install scikit-learn
pip install numpy<2.0.0 # deepspeed が1系のみ対応
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_UTILS=1 pip install deepspeed --no-cache-dir

pip install wheel packaging
MAX_JOBS=4 pip install flash-attn --no-build-isolation

pip install sudachipy
