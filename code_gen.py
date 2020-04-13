# this program basically does a constexpr and generates cuda code
import textwrap
import numpy as np
from code_fragments import *
from utils import *


import argparse
parser = argparse.ArgumentParser(description='CodeGen V1')

parser.add_argument('--A_dim', type=int, default=12)
parser.add_argument('--B_dim', type=int, default=12)
parser.add_argument('--C_dim', type=int, default=12)
parser.add_argument('--A_blocks', type=int, default=12)
parser.add_argument('--C_blocks', type=int, default=12)
parser.add_argument('--Gy', type=int, default=12)
parser.add_argument('--infile', default=None, type=str)
parser.add_argument('--infile_bias', default=None, type=str)
parser.add_argument('--outfile', default=None, type=str)
parser.add_argument('--in_format', default="NCHW",type=str)
parser.add_argument('--out_format', default="NCHW",type=str)
parser.add_argument('--Tsb',type=float,default=1)
parser.add_argument('--fuse',default=False,action='store_true')


args = parser.parse_args()
GY = args.Gy
FUSE_END = args.fuse
print(FUSE_END)
TSB_MULT = args.Tsb
A_dim = args.A_dim
B_dim = args.B_dim
C_dim = args.C_dim
A_blocks = args.A_blocks
C_blocks = args.C_blocks
input_file = args.infile
outfile = args.outfile
assert C_dim % C_blocks == 0
GSY = C_dim // C_blocks
TSB =int( GSY * TSB_MULT)

IN_FORMAT = args.in_format
OUT_FORMAT = args.out_format

input_file_bias = args.infile_bias
if input_file_bias:
    bias = np.load(input_file_bias)

if IN_FORMAT == "NCHW":
    LOAD_CACHE = """
    RC = BC[IDX + C_offset + lane];
    """
else:
    LOAD_CACHE = """
    RC = smem_cache[lane][IDX];
    """

MAIN_PROGRAM = """
ACC[IDX_1] += RC * VAL;   
"""

LOAD_SHARED = """
#if Gsy == 32
for(int j=0;j < 32;j ++)
{
    int C_idx = C_offset + j;
    int B_idx = TILE + lane;
    smem_cache[j][lane] = BC[C_idx * B_dim + B_idx];
}
#else
#pragma unroll
for(int j =0; j < Gsy; j++)
{
#pragma unroll
    for(int i = lane; i < TSB; i += Gsy)
    {
        int C_idx = C_offset + j;
        int B_idx = TILE + i;
        smem_cache[j][i] = BC[C_idx * B_dim + B_idx];
    }
}
#endif
#if Gsy > 32
__syncthreads();
#else
__syncwarp();
#endif
"""

if IN_FORMAT == "NCHW":
    def emit_load_block(B_idx):
        new_block = LOAD_CACHE.replace("IDX",str(B_idx * C_dim))
        return new_block
else:
    def emit_load_block(B_idx,B_offset):
        new_block = LOAD_CACHE.replace("IDX",str(B_idx - B_offset))
        return new_block

def emit_load_smem_block(local_TSB, tile_id):
    return LOAD_SHARED.replace("TSB",str(local_TSB)).replace("TILE",str(tile_id * TSB))

def emit_compute_block(Ny_idx,val):

    new_block = MAIN_PROGRAM.replace("IDX_1",str(Ny_idx)).replace("VAL",str(val)+"f")
    return new_block


def ny_to_a(ny_idx,groupId,blockId, A_dim = None, A_offset = None):
    if A_offset is None:
        A_offset = blockId * (A_dim // A_blocks)
    return A_offset + ny_idx

def generate_from_B(Ny_indices, B_indices,BA,block,NY,GY = None,A_offset=None):

    program = ""


    for group in range(GY):
        program += GROUP_CONTROL_START.replace("GROUP",str(group)) + "\n"

        next_tile_start = 0
        old_b_idx = -1
        for ny_idx, b_idx in zip(Ny_indices[group],B_indices[group]):

            if IN_FORMAT == "NHWC":
                if old_b_idx < next_tile_start and b_idx >= next_tile_start:
                    smem_block = emit_load_smem_block(min(TSB,B_dim - next_tile_start),next_tile_start // TSB)
                    program += textwrap.indent(smem_block,"\t")
                    next_tile_start += TSB

            if b_idx != old_b_idx:
                if IN_FORMAT == "NCHW":
                    load_block_cuda = emit_load_block(b_idx)
                else:
                    load_block_cuda = emit_load_block(b_idx,next_tile_start - TSB)
                program += textwrap.indent(load_block_cuda,"\t")
                old_b_idx = b_idx

            a_idx = ny_to_a(ny_idx,group,block,A_dim = A_dim, A_offset=A_offset)
            value = BA[b_idx,a_idx]

            compute_block_cuda = emit_compute_block(ny_idx,  value)
            program += textwrap.indent(compute_block_cuda, "\t")

        print(block,group)
        program += GROUP_CONTROL_END + "\n"

    return program


def get_idx_balanced(block,BA,A_offset,block_NY,GY=None):

    Ny_indices = [[] for i in range(GY)]
    B_indices = [[] for i in range(GY)]
    nnz = np.sum(np.abs(BA[:,A_offset:A_offset + block_NY]) > EPS )
    nnz_per_group = nnz // GY
    curr_group = 0
    curr_nnz = 0
    for B_idx in range(B_dim):
        for ny in range(block_NY):
            assert curr_group < GY
            A_idx = ny_to_a(ny,curr_group,block,A_dim = A_dim, A_offset=A_offset)
            if np.abs(BA[B_idx,A_idx]) > EPS:
                B_indices[curr_group].append(B_idx)
                Ny_indices[curr_group].append(ny)
                curr_nnz += 1
            if curr_nnz > nnz_per_group:
                curr_group += 1
                curr_nnz = 0

    return Ny_indices, B_indices

def no_load_balance(BA):

    assert A_dim % A_blocks == 0
    interval = A_dim // A_blocks
    bounds = [interval * i for i in range(A_blocks + 1)]
    return bounds , interval

def load_balancer2(BA):

    total_nnz = (np.abs(BA) > EPS).sum()
    nnz_per_block = total_nnz / A_blocks
    sums = np.sum(np.abs(BA) > EPS, axis = 0)
    cs = np.cumsum(sums)
    bounds = [np.argmax(cs > nnz_per_block * i) for i in range(A_blocks)]
    bounds = bounds + [A_dim]
    nnzs = np.diff(bounds)
    NY = np.max(nnzs)
    return bounds, NY


# name is the name of the numpy file
def gencode(BA,outfile,C_dim,A_blocks,C_blocks,GY,name=None):
    program = ""
    assert A_dim % A_blocks == 0
    assert C_dim % C_blocks == 0
    B_dim = BA.shape[0]

    if IN_FORMAT == "NCHW" and OUT_FORMAT == "NCHW":
        bounds, NY = load_balancer2(BA)
    else:
        bounds, NY = no_load_balance(BA)

    program += START_NONFUSED.replace("OUTPUT_FORMAT",OUT_FORMAT).replace("INPUT_FORMAT",IN_FORMAT).replace("ST_VAL",str(ST)).replace("Ny",str(NY)).replace("GY",str(GY)).replace("A_dim",str(A_dim)).replace(
        "C_dim",str(C_dim)).replace("B_dim",str(B_dim)).replace("A_BLOCKS",str(A_blocks)).replace("C_BLOCKS",str(C_blocks)).replace("TSB",str(TSB)) + "\n"

    for block in range(A_blocks):
    #for block in range(1):
        A_offset = bounds[block]
        block_NY = bounds[block+1] - A_offset
        program += BLOCK_CONTROL_START.replace("BLOCK", str(block)) + "\n"
        Ny_indices, B_indices = get_idx_balanced(block,BA,A_offset,block_NY,GY=GY)

        program += textwrap.indent(generate_from_B(Ny_indices,B_indices,BA,block,NY,GY=GY,A_offset=A_offset),"\t") + "\n"
        if OUT_FORMAT == "NCHW":
            if FUSE_END:
                if GY > 1:
                    print("End fusion strategy not valid.")
                for i in range(block_NY):
                    program += BLOCK_END_REDUCTION.replace("OFFSET",str((A_offset + i) * C_dim)).replace("IDX",str(i)).replace("BIAS",str(bias[A_offset+i]))
            else:
                program += BLOCK_END.replace("A_offset",str(A_offset)).replace("Ny",str(block_NY)).replace("A_BLOCKS",str(A_blocks)).replace(
            "C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)) + "\n"
        else:
            program += BLOCK_END_NHWC.replace("A_offset",str(A_offset)).replace("Ny",str(block_NY)).replace("A_BLOCKS",str(A_blocks)).replace(
                "C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)) + "\n"
        program += BLOCK_CONTROL_END

    program += END_NONFUSED.replace("A_BLOCKS",str(A_blocks)).replace("C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).\
        replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)).replace("AB_sparse_tidy.npy",name)
    open(outfile,"w").write(program.replace("B_dim",str(B_dim)))


BA = np.load(input_file)
print(BA.shape)
BA = BA.squeeze()

gencode(BA,outfile,C_dim,A_blocks,C_blocks,GY,name=input_file)
