import faiss
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--sample_ratio",type=float,default=0.3)
    parser.add_argument("--output_path",required=True)
    # build index argument
    # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    # dim should be a multiple of m, nlist just like ncentroids, nbits_per_idx is usually 8
    parser.add_argument("--nlist",type=int,default=32768)
    parser.add_argument("--m",type=int,default=16)
    parser.add_argument("--nbits_per_idx",type=int,default=8)
    args = parser.parse_args()

    embedding_files = [os.path.join(args.embedding_dir,x) for x in os.listdir(args.embedding_dir) if x.endswith("pt")]
    embedding_files.sort(key=lambda x:os.path.basename(x).split(".")[0].split("_")[-2:])

    embeddings_for_training = []
    for file in embedding_files:
        print("loading from ", file, flush=True)
        data = torch.load(file)
        sampled_data = data[torch.randint(0, high=data.size(0), size=(int(data.size(0) * args.sample_ratio),))]
        embeddings_for_training.append(sampled_data)

    embeddings_for_training = torch.cat(embeddings_for_training,dim=0)
    print(f"{embeddings_for_training.shape=}", flush=True)

    ## build index
    quantizer = faiss.IndexFlatL2(args.dim)
    index = faiss.IndexIVFPQ(quantizer, args.dim, args.nlist, args.m, args.nbits_per_idx)

    print("training", flush=True)
    ## training
    gpu_resource = faiss.StandardGpuResources()
    gpu_quantizer = faiss.index_cpu_to_gpu(gpu_resource, 0, quantizer)
    gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
    # faiss version:faiss-gpu=1.7.0, need numpy array
    embeddings_for_training = np.array(embeddings_for_training, dtype=np.float32)
    gpu_index.train(embeddings_for_training)

    ## add
    ## if OOM, try to split into small batches
    batch_size = 1000
    for file in tqdm(embedding_files,desc='loading from embedding files'):
        data = torch.load(file)
        data = np.array(data, dtype=np.float32)
        # Split data into batches
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            gpu_index.add(batch_data)

    cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    ## save
    faiss.write_index(cpu_index, args.output_path)
