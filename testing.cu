
#include <cnpy.h>

#include <vector>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (3136 / 98)
#define Gsy Tsz
#define Gy 1
#define Block_size (Gy * Gsy)
#define In_Format 'NCHW'
#define Out_Format 'NCHW'

namespace cg = cooperative_groups;

__global__ void mm(const float * __restrict__ BC, float * AC)
{
    register float ACC[64] = {0.0};
	register float RC = 0.0;
#if Gy > 1	
        __shared__ float result[64][Tsz];
	for(int i = threadIdx.x; i < 64 * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();
#endif
#if In_Format == 'NHWC'
	__shared__ float smem_cache[Tsz][32+1];
#endif
#if Out_Format == 'NHWC'
	__shared__ float smem_result[Tsz][64+1];
#endif

	int A_offset = blockIdx.x * (128 / 2);
	int C_offset = blockIdx.y * (3136 / 98);
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);


if(blockIdx.x == 0)
{



	if(groupId == 0)
	{


		    RC = BC[0 + C_offset + lane];
    
		ACC[1] += RC * -0.18584795f;   

		ACC[2] += RC * 0.07350932f;   

		ACC[4] += RC * 1.0029237f;   

		ACC[27] += RC * 0.16351075f;   

		ACC[28] += RC * 0.05196784f;   

		ACC[29] += RC * -0.11776184f;   

		ACC[45] += RC * 0.9202288f;   

		ACC[46] += RC * -0.10351631f;   

		ACC[52] += RC * -0.19278231f;   

		    RC = BC[3136 + C_offset + lane];
    
		ACC[8] += RC * 0.49481606f;   

		ACC[17] += RC * 0.102857f;   

		ACC[24] += RC * 0.34323132f;   

		ACC[28] += RC * 0.11447914f;   

		ACC[30] += RC * 1.0406339f;   

		ACC[38] += RC * 0.12667674f;   

		ACC[62] += RC * -0.105981514f;   

		    RC = BC[6272 + C_offset + lane];
    
		ACC[10] += RC * -0.21451224f;   

		ACC[21] += RC * 0.34443143f;   

		ACC[46] += RC * -0.11600338f;   

		    RC = BC[9408 + C_offset + lane];
    
		ACC[21] += RC * -0.54315495f;   

		ACC[22] += RC * -0.32288057f;   

		    RC = BC[12544 + C_offset + lane];
    
		ACC[2] += RC * 0.035814032f;   

		ACC[29] += RC * 0.15621199f;   

		ACC[32] += RC * -0.29438356f;   

		ACC[34] += RC * 0.19884212f;   

		    RC = BC[18816 + C_offset + lane];
    
		ACC[13] += RC * -0.4240484f;   

		ACC[35] += RC * -0.59533787f;   

		ACC[62] += RC * 0.14454196f;   

		    RC = BC[21952 + C_offset + lane];
    
		ACC[0] += RC * -0.22699068f;   

		ACC[6] += RC * 0.13832037f;   

		ACC[19] += RC * -0.27171835f;   

		ACC[29] += RC * -0.16838886f;   

		ACC[31] += RC * 0.18778998f;   

		ACC[41] += RC * 0.1515109f;   

		ACC[42] += RC * -0.645049f;   

		ACC[43] += RC * -0.66751015f;   

		ACC[47] += RC * -0.13399652f;   

		ACC[54] += RC * -0.14735857f;   

		ACC[60] += RC * -0.2250801f;   

		ACC[62] += RC * 0.10988758f;   

		ACC[63] += RC * 0.121927194f;   

		    RC = BC[25088 + C_offset + lane];
    
		ACC[32] += RC * 0.39211276f;   

		ACC[51] += RC * 0.64756685f;   

		    RC = BC[28224 + C_offset + lane];
    
		ACC[9] += RC * 0.23206377f;   

		ACC[11] += RC * 0.355418f;   

		ACC[38] += RC * 0.081785716f;   

		    RC = BC[31360 + C_offset + lane];
    
		ACC[0] += RC * 0.32356438f;   

		ACC[3] += RC * -0.6257694f;   

		ACC[18] += RC * -0.36066723f;   

		ACC[19] += RC * 0.19064696f;   

		ACC[29] += RC * -0.18113218f;   

		ACC[34] += RC * 0.21888024f;   

		ACC[42] += RC * 0.4985783f;   

		ACC[43] += RC * 0.33376634f;   

		ACC[44] += RC * 0.1442427f;   

		    RC = BC[34496 + C_offset + lane];
    
		ACC[1] += RC * -0.12532695f;   

		ACC[6] += RC * 0.15466522f;   

		ACC[8] += RC * -0.35663697f;   

		ACC[11] += RC * -0.8898302f;   

		ACC[20] += RC * -0.18002282f;   

		ACC[29] += RC * 0.37181646f;   

		ACC[41] += RC * 0.18639983f;   

		ACC[46] += RC * -0.037677366f;   

		    RC = BC[37632 + C_offset + lane];
    
		ACC[2] += RC * 0.04931372f;   

		ACC[29] += RC * 0.2322457f;   

		ACC[34] += RC * -0.40378276f;   

		ACC[40] += RC * 0.2730293f;   

		ACC[52] += RC * -0.19606051f;   

		    RC = BC[40768 + C_offset + lane];
    
		ACC[2] += RC * 0.033532735f;   

		ACC[3] += RC * -0.25232637f;   

		ACC[16] += RC * -0.33985403f;   

		ACC[18] += RC * -0.30187717f;   

		ACC[19] += RC * 0.24938029f;   

		ACC[46] += RC * 0.059925582f;   

		ACC[48] += RC * 0.19251324f;   

		ACC[54] += RC * -0.29366392f;   

		ACC[60] += RC * -0.34095109f;   

		    RC = BC[43904 + C_offset + lane];
    
		ACC[13] += RC * 0.3031487f;   

		ACC[25] += RC * -0.13325948f;   

		ACC[35] += RC * 0.7098635f;   

		ACC[44] += RC * 0.4846318f;   

		ACC[50] += RC * -0.8369425f;   

		ACC[57] += RC * 0.520863f;   

		    RC = BC[47040 + C_offset + lane];
    
		ACC[2] += RC * -0.043606438f;   

		ACC[4] += RC * 0.6640511f;   

		ACC[17] += RC * -0.12276016f;   

		ACC[20] += RC * -0.3290968f;   

		ACC[27] += RC * -0.13328187f;   

		ACC[29] += RC * -0.15515116f;   

		ACC[51] += RC * -0.34699997f;   

		ACC[52] += RC * 0.19347595f;   

		    RC = BC[50176 + C_offset + lane];
    
		ACC[31] += RC * -0.12047405f;   

		ACC[62] += RC * 0.22767809f;   

		ACC[63] += RC * -0.27400735f;   

		    RC = BC[53312 + C_offset + lane];
    
		ACC[9] += RC * -0.3624156f;   

		ACC[47] += RC * 0.104587376f;   

		ACC[48] += RC * -0.49369574f;   

		ACC[52] += RC * 0.31744704f;   

		ACC[56] += RC * -0.37888768f;   

		ACC[59] += RC * 0.27705458f;   

		ACC[61] += RC * -0.59206426f;   

		ACC[62] += RC * -0.16184911f;   

		ACC[63] += RC * -0.18657786f;   

		    RC = BC[56448 + C_offset + lane];
    
		ACC[14] += RC * 0.21536787f;   

		ACC[31] += RC * 0.49074826f;   

		ACC[53] += RC * -0.36852106f;   

		ACC[55] += RC * -0.53320783f;   

		ACC[58] += RC * -0.40168795f;   

		ACC[62] += RC * 0.22343624f;   

		    RC = BC[59584 + C_offset + lane];
    
		ACC[1] += RC * -0.15016143f;   

		ACC[2] += RC * 0.04522205f;   

		ACC[9] += RC * 0.10925903f;   

		ACC[10] += RC * -0.3644416f;   

		ACC[11] += RC * 0.2997704f;   

		ACC[15] += RC * 0.2818224f;   

		ACC[21] += RC * 1.2042669f;   

		ACC[46] += RC * -0.13913985f;   

		ACC[52] += RC * -0.19981512f;   

		    RC = BC[62720 + C_offset + lane];
    
		ACC[13] += RC * -0.42296624f;   

		ACC[44] += RC * 0.19960028f;   

		ACC[53] += RC * -0.39095968f;   

		ACC[62] += RC * -0.14741841f;   

		    RC = BC[65856 + C_offset + lane];
    
		ACC[2] += RC * 0.13015383f;   

		ACC[10] += RC * -0.27304852f;   

		ACC[15] += RC * 0.12982921f;   

		ACC[17] += RC * -0.09169744f;   

		ACC[22] += RC * -0.11175012f;   

		ACC[29] += RC * -0.17992558f;   

		ACC[38] += RC * -0.10121323f;   

		ACC[39] += RC * 0.05587235f;   

		ACC[41] += RC * 0.25168929f;   

		ACC[46] += RC * -0.1194533f;   

		    RC = BC[68992 + C_offset + lane];
    
		ACC[2] += RC * 0.037355147f;   

		ACC[29] += RC * 0.24902283f;   

		    RC = BC[72128 + C_offset + lane];
    
		ACC[15] += RC * -0.28222182f;   

		ACC[46] += RC * 0.03862857f;   

		    RC = BC[75264 + C_offset + lane];
    
		ACC[17] += RC * 0.13590872f;   

		ACC[23] += RC * 0.9363806f;   

		ACC[25] += RC * 0.09702684f;   

		ACC[28] += RC * 0.066035986f;   

		ACC[44] += RC * 0.18337922f;   

		    RC = BC[78400 + C_offset + lane];
    
		ACC[17] += RC * 0.08865731f;   

		ACC[49] += RC * -0.6772053f;   

		    RC = BC[81536 + C_offset + lane];
    
		ACC[1] += RC * 0.085653275f;   

		ACC[4] += RC * 0.44263017f;   

		ACC[20] += RC * -0.21628965f;   

		ACC[32] += RC * -0.38544872f;   

		ACC[34] += RC * 0.37335655f;   

		ACC[40] += RC * -0.3120128f;   

		    RC = BC[84672 + C_offset + lane];
    
		ACC[0] += RC * -0.20678516f;   

		ACC[3] += RC * 0.32530537f;   

		ACC[7] += RC * -0.64888483f;   

		ACC[16] += RC * 0.24505733f;   

		ACC[18] += RC * 0.5256636f;   

		ACC[19] += RC * -0.3272166f;   

		ACC[22] += RC * -0.13409023f;   

		ACC[26] += RC * -0.17373987f;   

		ACC[31] += RC * 0.2050952f;   

		ACC[47] += RC * 0.11039164f;   

		ACC[48] += RC * -0.2266204f;   

		ACC[54] += RC * -0.21862362f;   

		ACC[59] += RC * 0.12699634f;   

		    RC = BC[87808 + C_offset + lane];
    
		ACC[9] += RC * 0.15279768f;   

		ACC[11] += RC * 0.16959256f;   

		ACC[16] += RC * -0.42881632f;   

		ACC[59] += RC * 0.14944461f;   

		    RC = BC[90944 + C_offset + lane];
    
		ACC[3] += RC * 0.33154777f;   

		ACC[9] += RC * -0.17489237f;   

		ACC[34] += RC * -0.23416986f;   

		ACC[38] += RC * -0.11405346f;   

		ACC[41] += RC * -0.121796f;   

		ACC[47] += RC * 0.16470617f;   

		ACC[52] += RC * 0.21700798f;   

		ACC[54] += RC * 0.15729876f;   

		ACC[56] += RC * 0.23263544f;   

		ACC[59] += RC * -0.21623065f;   

		ACC[61] += RC * 0.60681957f;   

		ACC[62] += RC * -0.2807839f;   

		ACC[63] += RC * 0.13468657f;   

		    RC = BC[94080 + C_offset + lane];
    
		ACC[15] += RC * -0.28389496f;   

		ACC[22] += RC * -0.23826738f;   

		ACC[46] += RC * 0.08869904f;   

		    RC = BC[97216 + C_offset + lane];
    
		ACC[15] += RC * -0.101051666f;   

		ACC[21] += RC * -0.20413171f;   

		ACC[46] += RC * 0.0930694f;   

		    RC = BC[100352 + C_offset + lane];
    
		ACC[49] += RC * -0.46746957f;   

		    RC = BC[103488 + C_offset + lane];
    
		ACC[7] += RC * -0.23927209f;   

		ACC[14] += RC * 0.39993063f;   

		ACC[28] += RC * 0.06608304f;   

		ACC[31] += RC * 0.41690415f;   

		ACC[41] += RC * -0.16268021f;   

		ACC[47] += RC * -0.06665658f;   

		ACC[54] += RC * 0.3898219f;   

		ACC[58] += RC * -0.4811902f;   

		ACC[62] += RC * -0.20769393f;   

		    RC = BC[106624 + C_offset + lane];
    
		ACC[4] += RC * 0.23032154f;   

		ACC[9] += RC * 0.23267138f;   

		ACC[18] += RC * 0.17405392f;   

		ACC[19] += RC * -0.30409917f;   

		ACC[26] += RC * -0.3498056f;   

		ACC[33] += RC * -0.41797748f;   

		ACC[37] += RC * -0.14948401f;   

		ACC[40] += RC * -0.7528111f;   

		ACC[47] += RC * -0.12718853f;   

		ACC[59] += RC * 0.30002242f;   

		ACC[60] += RC * 0.23654762f;   

		    RC = BC[109760 + C_offset + lane];
    
		ACC[2] += RC * 0.029297013f;   

		ACC[26] += RC * 0.14118852f;   

		ACC[28] += RC * -0.09510913f;   

		ACC[33] += RC * 0.24073741f;   

		ACC[46] += RC * -0.07880714f;   

		    RC = BC[112896 + C_offset + lane];
    
		ACC[13] += RC * 0.47092745f;   

		ACC[25] += RC * 0.14386271f;   

		ACC[35] += RC * 0.6366757f;   

		ACC[44] += RC * -0.27373043f;   

		ACC[50] += RC * -0.7100229f;   

		ACC[57] += RC * 0.60796255f;   

		    RC = BC[116032 + C_offset + lane];
    
		ACC[8] += RC * 0.25282413f;   

		ACC[18] += RC * -0.106563374f;   

		ACC[28] += RC * -0.16593163f;   

		ACC[41] += RC * 0.30251354f;   

		ACC[47] += RC * 0.13658065f;   

		ACC[54] += RC * -0.69706804f;   

		    RC = BC[119168 + C_offset + lane];
    
		ACC[19] += RC * -0.1432965f;   

		ACC[31] += RC * 0.29604664f;   

		ACC[34] += RC * -0.25980312f;   

		ACC[38] += RC * 0.12999922f;   

		ACC[47] += RC * -0.0854889f;   

		ACC[49] += RC * -0.29304054f;   

		ACC[54] += RC * -0.27289852f;   

		ACC[56] += RC * 0.20387344f;   

		ACC[58] += RC * 0.23806095f;   

		ACC[60] += RC * -0.18487847f;   

		ACC[63] += RC * 0.38124996f;   

		    RC = BC[122304 + C_offset + lane];
    
		ACC[22] += RC * -0.32027736f;   

		    RC = BC[125440 + C_offset + lane];
    
		ACC[1] += RC * -0.1730987f;   

		ACC[2] += RC * 0.06058641f;   

		ACC[11] += RC * -0.47734895f;   

		ACC[20] += RC * -0.3253965f;   

		ACC[27] += RC * 0.23532234f;   

		ACC[29] += RC * 0.28981793f;   

		ACC[32] += RC * 0.27740094f;   

		ACC[38] += RC * 0.08028157f;   

		ACC[41] += RC * 0.2635885f;   

		ACC[43] += RC * -0.30340588f;   

		ACC[46] += RC * -0.06796846f;   

		ACC[51] += RC * -0.3080269f;   

		ACC[52] += RC * -0.3164784f;   

		ACC[54] += RC * -0.1577772f;   

		    RC = BC[128576 + C_offset + lane];
    
		ACC[1] += RC * 0.17775148f;   

		ACC[2] += RC * -0.052164525f;   

		ACC[17] += RC * -0.14576583f;   

		ACC[29] += RC * -0.30311817f;   

		ACC[38] += RC * -0.10872306f;   

		ACC[39] += RC * 0.08131772f;   

		ACC[41] += RC * -0.18844928f;   

		ACC[51] += RC * 0.16056864f;   

		    RC = BC[131712 + C_offset + lane];
    
		ACC[6] += RC * 0.13167883f;   

		ACC[17] += RC * 0.098335385f;   

		ACC[31] += RC * -0.37775412f;   

		ACC[38] += RC * 0.1992557f;   

		ACC[39] += RC * -0.029296694f;   

		ACC[41] += RC * 0.142547f;   

		ACC[47] += RC * -0.08436086f;   

		ACC[49] += RC * -0.31818718f;   

		ACC[54] += RC * -0.4348505f;   

		ACC[55] += RC * 0.5758083f;   

		ACC[58] += RC * -0.31610832f;   

		ACC[62] += RC * -0.28211477f;   

		ACC[63] += RC * -0.088721775f;   

		    RC = BC[134848 + C_offset + lane];
    
		ACC[1] += RC * 0.16146538f;   

		ACC[2] += RC * -0.0475772f;   

		ACC[6] += RC * 0.3263237f;   

		ACC[8] += RC * -0.6276081f;   

		ACC[9] += RC * 0.11742934f;   

		ACC[10] += RC * 0.20307045f;   

		ACC[17] += RC * 0.12657759f;   

		ACC[28] += RC * 0.10099993f;   

		ACC[29] += RC * -0.24853194f;   

		ACC[38] += RC * 0.13664165f;   

		ACC[39] += RC * -0.059467845f;   

		ACC[41] += RC * 0.27962637f;   

		ACC[47] += RC * -0.33069918f;   

		ACC[62] += RC * 0.4070311f;   

		    RC = BC[137984 + C_offset + lane];
    
		ACC[5] += RC * -0.5277919f;   

		ACC[13] += RC * 0.4151213f;   

		ACC[23] += RC * 0.3098378f;   

		ACC[50] += RC * 0.60081923f;   

		ACC[53] += RC * 0.9438657f;   

		ACC[57] += RC * -0.48226568f;   

		ACC[62] += RC * -0.12101145f;   

		    RC = BC[141120 + C_offset + lane];
    
		ACC[1] += RC * -0.22886378f;   

		ACC[2] += RC * 0.12361719f;   

		ACC[4] += RC * -0.72761476f;   

		ACC[20] += RC * 0.07841185f;   

		ACC[27] += RC * 0.1727555f;   

		ACC[29] += RC * 0.22075741f;   

		ACC[32] += RC * 0.24208428f;   

		ACC[38] += RC * 0.18292496f;   

		ACC[41] += RC * 0.30513752f;   

		ACC[45] += RC * -0.2635817f;   

		ACC[46] += RC * -0.0830069f;   

		ACC[52] += RC * -0.44706273f;   

		ACC[54] += RC * -0.17014933f;   

		    RC = BC[144256 + C_offset + lane];
    
		ACC[12] += RC * 0.2964925f;   

		ACC[29] += RC * 0.26264462f;   

		ACC[32] += RC * 0.14424336f;   

		    RC = BC[147392 + C_offset + lane];
    
		ACC[16] += RC * 0.27318072f;   

		ACC[18] += RC * -0.17249444f;   

		ACC[19] += RC * 0.1799758f;   

		ACC[42] += RC * 0.37337103f;   

		ACC[54] += RC * -0.13773881f;   

		ACC[60] += RC * -0.15917754f;   

		    RC = BC[150528 + C_offset + lane];
    
		ACC[24] += RC * 0.8332341f;   

		ACC[25] += RC * -0.12432293f;   

		ACC[30] += RC * -1.3509405f;   

		ACC[44] += RC * 0.37730205f;   

		    RC = BC[153664 + C_offset + lane];
    
		ACC[17] += RC * -0.080590144f;   

		ACC[28] += RC * -0.067727126f;   

		ACC[38] += RC * -0.11767121f;   

		ACC[47] += RC * -0.053688377f;   

		    RC = BC[156800 + C_offset + lane];
    
		ACC[6] += RC * 0.17148225f;   

		ACC[19] += RC * 0.15576942f;   

		ACC[34] += RC * 0.24993496f;   

		ACC[43] += RC * 0.28970215f;   

		ACC[47] += RC * -0.049092744f;   

		ACC[48] += RC * -0.2749422f;   

		ACC[49] += RC * -0.482139f;   

		ACC[54] += RC * -0.21819173f;   

		ACC[56] += RC * -0.2958262f;   

		ACC[60] += RC * 0.1939095f;   

		    RC = BC[159936 + C_offset + lane];
    
		ACC[11] += RC * -0.20927271f;   

		ACC[16] += RC * -0.47554716f;   

		    RC = BC[163072 + C_offset + lane];
    
		ACC[25] += RC * -0.04016468f;   

		ACC[26] += RC * 0.23068126f;   

		ACC[33] += RC * 0.4181578f;   

		ACC[36] += RC * -1.1136609f;   

		ACC[37] += RC * 0.097754344f;   

		ACC[60] += RC * -0.32576373f;   

		    RC = BC[166208 + C_offset + lane];
    
		ACC[13] += RC * -0.30980462f;   

		ACC[23] += RC * -0.4268186f;   

		ACC[50] += RC * -0.28890938f;   

		ACC[57] += RC * 0.23223265f;   

		    RC = BC[172480 + C_offset + lane];
    
		ACC[1] += RC * -0.16658776f;   

		ACC[2] += RC * 0.0996416f;   

		ACC[9] += RC * 0.19291757f;   

		ACC[11] += RC * 0.63438296f;   

		ACC[20] += RC * -0.1459365f;   

		ACC[27] += RC * 0.11957474f;   

		ACC[29] += RC * -0.24668978f;   

		ACC[46] += RC * -0.08030703f;   

		ACC[52] += RC * -0.3264767f;   

		ACC[60] += RC * -0.21778284f;   

		    RC = BC[175616 + C_offset + lane];
    
		ACC[17] += RC * 0.19204687f;   

		ACC[38] += RC * 0.1554395f;   

		ACC[39] += RC * -0.047485f;   

		ACC[41] += RC * 0.11880568f;   

		ACC[54] += RC * -0.23030247f;   

		    RC = BC[178752 + C_offset + lane];
    
		ACC[12] += RC * -0.66184276f;   

		ACC[32] += RC * -0.33932766f;   

		ACC[42] += RC * -0.2929955f;   

		ACC[51] += RC * 0.20616594f;   

		ACC[52] += RC * 0.1678827f;   

		    RC = BC[181888 + C_offset + lane];
    
		ACC[2] += RC * 0.043896176f;   

		ACC[22] += RC * -0.20326512f;   

		ACC[26] += RC * -0.21282434f;   

		ACC[28] += RC * -0.08408172f;   

		ACC[33] += RC * -0.3274323f;   

		ACC[54] += RC * -0.23184451f;   

		ACC[60] += RC * 0.25018582f;   

		    RC = BC[185024 + C_offset + lane];
    
		ACC[15] += RC * -0.23927557f;   

		ACC[22] += RC * -0.112599134f;   

		ACC[46] += RC * 0.10714278f;   

		    RC = BC[188160 + C_offset + lane];
    
		ACC[2] += RC * -0.06315396f;   

		ACC[9] += RC * 0.3384536f;   

		ACC[22] += RC * 0.15122199f;   

		ACC[29] += RC * -0.27005216f;   

		ACC[31] += RC * -0.11214516f;   

		ACC[41] += RC * 0.27858534f;   

		ACC[52] += RC * -0.15154484f;   

		ACC[54] += RC * -0.2562287f;   

		ACC[63] += RC * -0.10051341f;   

		    RC = BC[191296 + C_offset + lane];
    
		ACC[2] += RC * -0.07809353f;   

		ACC[6] += RC * 0.12629469f;   

		ACC[10] += RC * 0.25280967f;   

		ACC[15] += RC * -0.36020976f;   

		ACC[21] += RC * 0.6824156f;   

		ACC[29] += RC * -0.18479209f;   

		ACC[41] += RC * -0.10888942f;   

		ACC[46] += RC * 0.12745552f;   

		ACC[52] += RC * 0.16152337f;   

		ACC[54] += RC * 0.13135277f;   

		    RC = BC[194432 + C_offset + lane];
    
		ACC[2] += RC * 0.020466007f;   

		ACC[6] += RC * -0.27377614f;   

		ACC[7] += RC * 0.1725072f;   

		ACC[21] += RC * 0.7983611f;   

		ACC[28] += RC * -0.088834435f;   

		ACC[29] += RC * 0.3583734f;   

		ACC[38] += RC * -0.08311054f;   

		ACC[52] += RC * -0.3431529f;   

		ACC[54] += RC * -0.22426634f;   

		    RC = BC[197568 + C_offset + lane];
    
		ACC[1] += RC * 0.13151796f;   

		ACC[17] += RC * -0.07872371f;   

		ACC[28] += RC * -0.08544529f;   

		ACC[39] += RC * 0.08887849f;   

	}



        AC[0 + C_offset  + lane] = max(ACC[0] + 3.977744f,0.0f);

        AC[3136 + C_offset  + lane] = max(ACC[1] + 6.530452f,0.0f);

        AC[6272 + C_offset  + lane] = max(ACC[2] + 0.6891806f,0.0f);

        AC[9408 + C_offset  + lane] = max(ACC[3] + 1.9361705f,0.0f);

        AC[12544 + C_offset  + lane] = max(ACC[4] + -3.6765223f,0.0f);

        AC[15680 + C_offset  + lane] = max(ACC[5] + 5.3994093f,0.0f);

        AC[18816 + C_offset  + lane] = max(ACC[6] + 2.343494f,0.0f);

        AC[21952 + C_offset  + lane] = max(ACC[7] + 4.982128f,0.0f);

        AC[25088 + C_offset  + lane] = max(ACC[8] + -0.9497072f,0.0f);

        AC[28224 + C_offset  + lane] = max(ACC[9] + -0.8413665f,0.0f);

        AC[31360 + C_offset  + lane] = max(ACC[10] + 10.717709f,0.0f);

        AC[34496 + C_offset  + lane] = max(ACC[11] + -0.6981136f,0.0f);

        AC[37632 + C_offset  + lane] = max(ACC[12] + 2.9473062f,0.0f);

        AC[40768 + C_offset  + lane] = max(ACC[13] + -1.6245604f,0.0f);

        AC[43904 + C_offset  + lane] = max(ACC[14] + 0.061071157f,0.0f);

        AC[47040 + C_offset  + lane] = max(ACC[15] + -2.6851745f,0.0f);

        AC[50176 + C_offset  + lane] = max(ACC[16] + 2.9532514f,0.0f);

        AC[53312 + C_offset  + lane] = max(ACC[17] + 2.908723f,0.0f);

        AC[56448 + C_offset  + lane] = max(ACC[18] + 0.58572805f,0.0f);

        AC[59584 + C_offset  + lane] = max(ACC[19] + 2.4610727f,0.0f);

        AC[62720 + C_offset  + lane] = max(ACC[20] + 5.1531906f,0.0f);

        AC[65856 + C_offset  + lane] = max(ACC[21] + -21.59833f,0.0f);

        AC[68992 + C_offset  + lane] = max(ACC[22] + 6.769128f,0.0f);

        AC[72128 + C_offset  + lane] = max(ACC[23] + -1.4368893f,0.0f);

        AC[75264 + C_offset  + lane] = max(ACC[24] + -2.4415185f,0.0f);

        AC[78400 + C_offset  + lane] = max(ACC[25] + 2.3887634f,0.0f);

        AC[81536 + C_offset  + lane] = max(ACC[26] + 4.2395906f,0.0f);

        AC[84672 + C_offset  + lane] = max(ACC[27] + 1.7414098f,0.0f);

        AC[87808 + C_offset  + lane] = max(ACC[28] + 4.020364f,0.0f);

        AC[90944 + C_offset  + lane] = max(ACC[29] + 3.3077111f,0.0f);

        AC[94080 + C_offset  + lane] = max(ACC[30] + 1.8318611f,0.0f);

        AC[97216 + C_offset  + lane] = max(ACC[31] + -3.4719281f,0.0f);

        AC[100352 + C_offset  + lane] = max(ACC[32] + 1.4029231f,0.0f);

        AC[103488 + C_offset  + lane] = max(ACC[33] + 2.0068154f,0.0f);

        AC[106624 + C_offset  + lane] = max(ACC[34] + 0.9218654f,0.0f);

        AC[109760 + C_offset  + lane] = max(ACC[35] + -6.0270934f,0.0f);

        AC[112896 + C_offset  + lane] = max(ACC[36] + 3.3614318f,0.0f);

        AC[116032 + C_offset  + lane] = max(ACC[37] + 2.8095539f,0.0f);

        AC[119168 + C_offset  + lane] = max(ACC[38] + 3.403461f,0.0f);

        AC[122304 + C_offset  + lane] = max(ACC[39] + 8.1054325f,0.0f);

        AC[125440 + C_offset  + lane] = max(ACC[40] + 3.8145614f,0.0f);

        AC[128576 + C_offset  + lane] = max(ACC[41] + -7.17782f,0.0f);

        AC[131712 + C_offset  + lane] = max(ACC[42] + 1.0847245f,0.0f);

        AC[134848 + C_offset  + lane] = max(ACC[43] + 1.5190965f,0.0f);

        AC[137984 + C_offset  + lane] = max(ACC[44] + -1.1734383f,0.0f);

        AC[141120 + C_offset  + lane] = max(ACC[45] + 0.07577264f,0.0f);

        AC[144256 + C_offset  + lane] = max(ACC[46] + 7.681935f,0.0f);

        AC[147392 + C_offset  + lane] = max(ACC[47] + 4.0016623f,0.0f);

        AC[150528 + C_offset  + lane] = max(ACC[48] + 4.7618046f,0.0f);

        AC[153664 + C_offset  + lane] = max(ACC[49] + 8.247117f,0.0f);

        AC[156800 + C_offset  + lane] = max(ACC[50] + 10.89353f,0.0f);

        AC[159936 + C_offset  + lane] = max(ACC[51] + 1.5308125f,0.0f);

        AC[163072 + C_offset  + lane] = max(ACC[52] + 5.9652867f,0.0f);

        AC[166208 + C_offset  + lane] = max(ACC[53] + 2.222279f,0.0f);

        AC[169344 + C_offset  + lane] = max(ACC[54] + 16.071999f,0.0f);

        AC[172480 + C_offset  + lane] = max(ACC[55] + 4.40592f,0.0f);

        AC[175616 + C_offset  + lane] = max(ACC[56] + 3.53694f,0.0f);

        AC[178752 + C_offset  + lane] = max(ACC[57] + -6.28261f,0.0f);

        AC[181888 + C_offset  + lane] = max(ACC[58] + 4.730913f,0.0f);

        AC[185024 + C_offset  + lane] = max(ACC[59] + 0.80060506f,0.0f);

        AC[188160 + C_offset  + lane] = max(ACC[60] + 4.3881507f,0.0f);

        AC[191296 + C_offset  + lane] = max(ACC[61] + 2.3128614f,0.0f);

        AC[194432 + C_offset  + lane] = max(ACC[62] + 4.072897f,0.0f);

        AC[197568 + C_offset  + lane] = max(ACC[63] + 4.2837954f,0.0f);

}

if(blockIdx.x == 1)
{



	if(groupId == 0)
	{


		    RC = BC[0 + C_offset + lane];
    
		ACC[62] += RC * -0.4406303f;   

		    RC = BC[3136 + C_offset + lane];
    
		ACC[22] += RC * 0.091088176f;   

		ACC[36] += RC * 0.55848783f;   

		ACC[46] += RC * 0.20598467f;   

		ACC[50] += RC * 0.22028092f;   

		ACC[53] += RC * 0.12283251f;   

		ACC[54] += RC * 0.13666105f;   

		    RC = BC[6272 + C_offset + lane];
    
		ACC[5] += RC * 0.20815681f;   

		ACC[18] += RC * -0.16649343f;   

		ACC[19] += RC * -0.5775647f;   

		ACC[24] += RC * 0.18504068f;   

		ACC[27] += RC * -0.57794267f;   

		ACC[38] += RC * -0.18265578f;   

		ACC[42] += RC * -0.2359908f;   

		ACC[44] += RC * 0.26388422f;   

		ACC[53] += RC * -0.08718044f;   

		ACC[63] += RC * 0.20574354f;   

		    RC = BC[9408 + C_offset + lane];
    
		ACC[8] += RC * 1.0667694f;   

		ACC[16] += RC * -0.12162542f;   

		ACC[61] += RC * -0.24287847f;   

		ACC[63] += RC * 0.08633752f;   

		    RC = BC[12544 + C_offset + lane];
    
		ACC[3] += RC * -0.6231197f;   

		ACC[38] += RC * -0.41663548f;   

		ACC[48] += RC * 0.13058908f;   

		    RC = BC[18816 + C_offset + lane];
    
		ACC[15] += RC * -0.08632509f;   

		    RC = BC[21952 + C_offset + lane];
    
		ACC[1] += RC * 0.14340843f;   

		ACC[4] += RC * -0.30484268f;   

		ACC[6] += RC * 0.36825827f;   

		ACC[10] += RC * 0.1624523f;   

		ACC[18] += RC * -0.32090935f;   

		ACC[21] += RC * -0.061324574f;   

		ACC[23] += RC * -0.15628628f;   

		ACC[24] += RC * 0.21420263f;   

		ACC[34] += RC * 0.3754557f;   

		ACC[54] += RC * 0.07404963f;   

		    RC = BC[25088 + C_offset + lane];
    
		ACC[12] += RC * 0.40567666f;   

		ACC[17] += RC * 0.23251288f;   

		ACC[30] += RC * -0.3227378f;   

		ACC[36] += RC * 0.2640671f;   

		ACC[41] += RC * -0.896036f;   

		ACC[59] += RC * -0.12072415f;   

		ACC[60] += RC * 0.32684088f;   

		    RC = BC[28224 + C_offset + lane];
    
		ACC[5] += RC * -0.52059394f;   

		ACC[11] += RC * 0.18056181f;   

		ACC[17] += RC * -0.36361873f;   

		ACC[18] += RC * 0.26469594f;   

		ACC[19] += RC * 0.5603244f;   

		ACC[24] += RC * -0.48774025f;   

		ACC[38] += RC * 0.2895176f;   

		ACC[42] += RC * 0.16287552f;   

		ACC[44] += RC * -0.8786358f;   

		ACC[53] += RC * 0.052632283f;   

		    RC = BC[31360 + C_offset + lane];
    
		ACC[1] += RC * 0.25844523f;   

		ACC[12] += RC * 0.3678633f;   

		ACC[18] += RC * -0.20652992f;   

		ACC[21] += RC * 0.02860625f;   

		ACC[23] += RC * 0.20161776f;   

		ACC[49] += RC * 0.64013994f;   

		    RC = BC[34496 + C_offset + lane];
    
		ACC[5] += RC * 0.33973366f;   

		ACC[18] += RC * -0.28541043f;   

		ACC[19] += RC * -0.5211075f;   

		ACC[24] += RC * 0.28496334f;   

		ACC[38] += RC * 0.4118586f;   

		ACC[44] += RC * 0.34815535f;   

		ACC[53] += RC * 0.07106397f;   

		    RC = BC[37632 + C_offset + lane];
    
		ACC[3] += RC * -0.6908116f;   

		ACC[38] += RC * -0.4022358f;   

		ACC[48] += RC * 0.18182096f;   

		    RC = BC[40768 + C_offset + lane];
    
		ACC[2] += RC * -0.27051896f;   

		ACC[4] += RC * 0.26418802f;   

		ACC[6] += RC * -0.22417636f;   

		ACC[13] += RC * 0.11076752f;   

		ACC[16] += RC * -0.2352699f;   

		ACC[22] += RC * -0.06142882f;   

		ACC[25] += RC * -0.23517811f;   

		ACC[28] += RC * 0.33756998f;   

		ACC[37] += RC * 0.3553626f;   

		ACC[39] += RC * -0.3641682f;   

		ACC[48] += RC * 0.38881904f;   

		ACC[57] += RC * 0.066223145f;   

		    RC = BC[43904 + C_offset + lane];
    
		ACC[30] += RC * 0.4237843f;   

		ACC[33] += RC * 0.45383757f;   

		ACC[36] += RC * -0.36284006f;   

		    RC = BC[47040 + C_offset + lane];
    
		ACC[41] += RC * 0.84823585f;   

		ACC[44] += RC * -0.20835146f;   

		ACC[60] += RC * -0.2641248f;   

		    RC = BC[50176 + C_offset + lane];
    
		ACC[14] += RC * -0.3333359f;   

		ACC[21] += RC * -0.18576123f;   

		ACC[29] += RC * 0.27656603f;   

		ACC[31] += RC * -0.09049547f;   

		ACC[43] += RC * -0.601761f;   

		ACC[57] += RC * -0.14112867f;   

		    RC = BC[53312 + C_offset + lane];
    
		ACC[2] += RC * 0.40836215f;   

		ACC[11] += RC * -0.12327007f;   

		ACC[17] += RC * 0.7166563f;   

		ACC[21] += RC * 0.14835687f;   

		ACC[23] += RC * 0.24019587f;   

		ACC[24] += RC * 0.23985353f;   

		ACC[34] += RC * -0.44319856f;   

		ACC[36] += RC * 0.49440685f;   

		ACC[40] += RC * 0.19167854f;   

		ACC[41] += RC * 0.307752f;   

		ACC[49] += RC * 0.26803735f;   

		    RC = BC[56448 + C_offset + lane];
    
		ACC[7] += RC * 0.6488399f;   

		ACC[11] += RC * 0.1301568f;   

		ACC[28] += RC * 0.30882934f;   

		ACC[29] += RC * -0.27747446f;   

		ACC[31] += RC * -0.22677533f;   

		ACC[57] += RC * 0.2715751f;   

		    RC = BC[59584 + C_offset + lane];
    
		ACC[5] += RC * -0.22435893f;   

		ACC[6] += RC * 0.41122705f;   

		ACC[11] += RC * 0.3052933f;   

		ACC[16] += RC * -0.15971515f;   

		ACC[19] += RC * 0.5101216f;   

		ACC[22] += RC * -0.067366935f;   

		ACC[24] += RC * -0.25120124f;   

		ACC[38] += RC * 0.14035158f;   

		ACC[40] += RC * -0.43715397f;   

		ACC[42] += RC * -0.17980106f;   

		ACC[51] += RC * 0.24254951f;   

		ACC[53] += RC * -0.07539134f;   

		ACC[63] += RC * 0.27311078f;   

		    RC = BC[62720 + C_offset + lane];
    
		ACC[26] += RC * 1.0373975f;   

		    RC = BC[65856 + C_offset + lane];
    
		ACC[1] += RC * 0.20897873f;   

		ACC[5] += RC * 0.47787747f;   

		ACC[13] += RC * -0.14578928f;   

		ACC[16] += RC * -0.1745981f;   

		ACC[18] += RC * -0.31983212f;   

		ACC[19] += RC * -0.94006985f;   

		ACC[22] += RC * -0.07130075f;   

		ACC[24] += RC * 0.38313714f;   

		ACC[27] += RC * -0.67933214f;   

		ACC[38] += RC * -0.18523245f;   

		ACC[42] += RC * -0.32479095f;   

		ACC[44] += RC * 0.31064206f;   

		ACC[48] += RC * 0.19336684f;   

		ACC[51] += RC * 0.57325405f;   

		ACC[53] += RC * -0.07721908f;   

		ACC[63] += RC * 0.20828615f;   

		    RC = BC[68992 + C_offset + lane];
    
		ACC[3] += RC * -0.44929525f;   

		ACC[38] += RC * -0.4956f;   

		ACC[48] += RC * 0.14596836f;   

		ACC[60] += RC * 0.16259946f;   

		    RC = BC[72128 + C_offset + lane];
    
		ACC[38] += RC * -0.23179027f;   

		ACC[40] += RC * 0.42678714f;   

		    RC = BC[75264 + C_offset + lane];
    
		ACC[15] += RC * -0.19233654f;   

		ACC[22] += RC * 0.07907943f;   

		ACC[46] += RC * -0.32640454f;   

		ACC[53] += RC * 0.02808538f;   

		ACC[56] += RC * 0.9160236f;   

		ACC[61] += RC * 0.15105776f;   

		    RC = BC[78400 + C_offset + lane];
    
		ACC[9] += RC * 0.36965203f;   

		ACC[33] += RC * 0.37337646f;   

		ACC[47] += RC * -0.13820614f;   

		ACC[50] += RC * 0.17192228f;   

		ACC[59] += RC * 0.1240179f;   

		    RC = BC[81536 + C_offset + lane];
    
		ACC[59] += RC * 0.08171398f;   

		    RC = BC[84672 + C_offset + lane];
    
		ACC[2] += RC * 0.5051878f;   

		ACC[4] += RC * -0.37206277f;   

		ACC[13] += RC * 0.1164334f;   

		ACC[16] += RC * -0.24667653f;   

		ACC[17] += RC * 0.25501922f;   

		ACC[25] += RC * 0.37876078f;   

		ACC[37] += RC * -0.30279717f;   

		ACC[39] += RC * -0.21880925f;   

		ACC[48] += RC * 0.32877222f;   

		ACC[49] += RC * -0.1759692f;   

		ACC[51] += RC * 0.15132824f;   

		ACC[52] += RC * 0.18719979f;   

		ACC[57] += RC * -0.074623965f;   

		    RC = BC[87808 + C_offset + lane];
    
		ACC[13] += RC * -0.07814742f;   

		ACC[24] += RC * -0.15904282f;   

		ACC[35] += RC * -1.0172532f;   

		ACC[38] += RC * -0.11877334f;   

		ACC[44] += RC * -0.315153f;   

		ACC[55] += RC * 0.26997954f;   

		    RC = BC[90944 + C_offset + lane];
    
		ACC[10] += RC * -0.38531703f;   

		ACC[11] += RC * -0.18128674f;   

		ACC[12] += RC * -0.4721898f;   

		ACC[18] += RC * 0.2796878f;   

		ACC[21] += RC * -0.053722873f;   

		ACC[23] += RC * -0.24814184f;   

		ACC[36] += RC * -0.2760433f;   

		ACC[37] += RC * 0.33000097f;   

		ACC[40] += RC * 0.15179594f;   

		ACC[47] += RC * 0.078985415f;   

		ACC[54] += RC * -0.12895069f;   

		    RC = BC[94080 + C_offset + lane];
    
		ACC[38] += RC * -0.23604664f;   

		ACC[40] += RC * 0.44253945f;   

		    RC = BC[97216 + C_offset + lane];
    
		ACC[6] += RC * 0.20294246f;   

		ACC[32] += RC * -0.47254777f;   

		ACC[38] += RC * -0.16003929f;   

		ACC[40] += RC * 0.30337873f;   

		ACC[61] += RC * 0.1879968f;   

		    RC = BC[100352 + C_offset + lane];
    
		ACC[9] += RC * 0.7538701f;   

		ACC[47] += RC * 0.09530357f;   

		    RC = BC[103488 + C_offset + lane];
    
		ACC[7] += RC * 0.530809f;   

		ACC[11] += RC * -0.16450137f;   

		ACC[14] += RC * -0.15800846f;   

		ACC[22] += RC * 0.063439f;   

		ACC[28] += RC * 0.35694954f;   

		ACC[29] += RC * -0.38846496f;   

		ACC[31] += RC * -0.20685743f;   

		ACC[39] += RC * 0.21267773f;   

		ACC[48] += RC * -0.19268388f;   

		ACC[52] += RC * 0.3178293f;   

		ACC[53] += RC * 0.0495657f;   

		ACC[54] += RC * 0.076075f;   

		ACC[57] += RC * 0.28270325f;   

		    RC = BC[106624 + C_offset + lane];
    
		ACC[18] += RC * 0.14951068f;   

		ACC[23] += RC * 0.17971806f;   

		ACC[24] += RC * -0.28865936f;   

		ACC[25] += RC * 0.37774938f;   

		ACC[37] += RC * -0.45645836f;   

		ACC[44] += RC * -0.19800922f;   

		ACC[56] += RC * -0.2593176f;   

		ACC[61] += RC * -0.13615313f;   

		    RC = BC[109760 + C_offset + lane];
    
		ACC[6] += RC * -0.4499957f;   

		ACC[8] += RC * 1.1393102f;   

		ACC[42] += RC * -0.13429818f;   

		ACC[48] += RC * 0.21427943f;   

		ACC[51] += RC * 0.26797965f;   

		ACC[53] += RC * -0.043698937f;   

		ACC[63] += RC * 0.096868806f;   

		    RC = BC[112896 + C_offset + lane];
    
		ACC[15] += RC * -0.12092155f;   

		ACC[26] += RC * -0.367689f;   

		ACC[33] += RC * 0.4610455f;   

		    RC = BC[116032 + C_offset + lane];
    
		ACC[10] += RC * 0.48143354f;   

		ACC[11] += RC * 0.1927379f;   

		ACC[13] += RC * 0.11634239f;   

		ACC[22] += RC * -0.13136567f;   

		ACC[39] += RC * -0.45201075f;   

		ACC[46] += RC * -0.14425245f;   

		ACC[48] += RC * 0.2557311f;   

		ACC[51] += RC * 0.33814302f;   

		ACC[53] += RC * -0.08074084f;   

		ACC[54] += RC * -0.16093913f;   

		ACC[63] += RC * 0.10440518f;   

		    RC = BC[119168 + C_offset + lane];
    
		ACC[9] += RC * -0.3006763f;   

		ACC[10] += RC * 0.34636855f;   

		ACC[11] += RC * 0.20608762f;   

		ACC[14] += RC * 0.24260342f;   

		ACC[17] += RC * -0.2925317f;   

		ACC[23] += RC * -0.20316848f;   

		ACC[43] += RC * 0.49689543f;   

		ACC[47] += RC * -0.22097862f;   

		ACC[52] += RC * -0.36790925f;   

		ACC[54] += RC * 0.11069383f;   

		    RC = BC[122304 + C_offset + lane];
    
		ACC[15] += RC * -0.08760723f;   

		ACC[32] += RC * -0.98680353f;   

		ACC[55] += RC * -0.6717163f;   

		ACC[61] += RC * -0.12629263f;   

		    RC = BC[125440 + C_offset + lane];
    
		ACC[0] += RC * -0.049236752f;   

		ACC[5] += RC * 0.31478804f;   

		ACC[12] += RC * -0.3559452f;   

		ACC[18] += RC * -0.2483207f;   

		ACC[19] += RC * -0.4347266f;   

		ACC[24] += RC * 0.32147488f;   

		ACC[42] += RC * -0.18268614f;   

		ACC[44] += RC * 0.5779853f;   

		ACC[48] += RC * 0.14837566f;   

		ACC[50] += RC * 0.15541449f;   

		ACC[53] += RC * 0.054397684f;   

		    RC = BC[128576 + C_offset + lane];
    
		ACC[1] += RC * -0.22827059f;   

		ACC[18] += RC * 0.16604872f;   

		ACC[22] += RC * -0.06136018f;   

		ACC[38] += RC * -0.21517621f;   

		ACC[50] += RC * -0.23421668f;   

		ACC[54] += RC * -0.09610469f;   

		    RC = BC[131712 + C_offset + lane];
    
		ACC[9] += RC * -0.45111597f;   

		ACC[10] += RC * 0.32330972f;   

		ACC[11] += RC * 0.25917482f;   

		ACC[14] += RC * -0.2415725f;   

		ACC[29] += RC * 0.28811523f;   

		ACC[47] += RC * -0.28767684f;   

		ACC[48] += RC * 0.1372894f;   

		ACC[50] += RC * 0.25730097f;   

		ACC[54] += RC * 0.1553135f;   

		ACC[59] += RC * 0.12625325f;   

		    RC = BC[134848 + C_offset + lane];
    
		ACC[6] += RC * -0.26206163f;   

		ACC[13] += RC * -0.13643606f;   

		ACC[14] += RC * -0.16992103f;   

		ACC[16] += RC * 0.23800203f;   

		ACC[21] += RC * -0.09796318f;   

		ACC[22] += RC * 0.055083614f;   

		ACC[42] += RC * 0.14952502f;   

		ACC[47] += RC * -0.29935917f;   

		ACC[48] += RC * -0.18766308f;   

		ACC[51] += RC * -0.31803492f;   

		ACC[61] += RC * -0.0584118f;   

		ACC[62] += RC * -0.6801283f;   

		    RC = BC[137984 + C_offset + lane];
    
		ACC[26] += RC * -0.38279307f;   

		ACC[33] += RC * 0.4642472f;   

		ACC[43] += RC * 0.48076943f;   

		ACC[45] += RC * 0.8154967f;   

		ACC[46] += RC * -0.49401584f;   

		ACC[56] += RC * -0.33043376f;   

		    RC = BC[141120 + C_offset + lane];
    
		ACC[0] += RC * -0.08347109f;   

		ACC[10] += RC * 0.27885252f;   

		ACC[11] += RC * 0.25426716f;   

		ACC[48] += RC * 0.26862422f;   

		ACC[53] += RC * 0.069793165f;   

		ACC[54] += RC * 0.112700365f;   

		    RC = BC[144256 + C_offset + lane];
    
		ACC[38] += RC * -0.39415953f;   

		ACC[59] += RC * -0.08427734f;   

		    RC = BC[147392 + C_offset + lane];
    
		ACC[1] += RC * -0.13147347f;   

		ACC[2] += RC * -0.23494175f;   

		ACC[4] += RC * 0.250107f;   

		ACC[13] += RC * 0.09344289f;   

		ACC[15] += RC * -0.19824103f;   

		ACC[28] += RC * 0.21619697f;   

		ACC[30] += RC * -0.5704185f;   

		ACC[31] += RC * -0.104502946f;   

		ACC[57] += RC * 0.051604852f;   

		ACC[61] += RC * 0.22274266f;   

		    RC = BC[150528 + C_offset + lane];
    
		ACC[30] += RC * 0.45899162f;   

		ACC[36] += RC * -0.5797182f;   

		    RC = BC[153664 + C_offset + lane];
    
		ACC[1] += RC * 0.15612988f;   

		ACC[5] += RC * -0.29628477f;   

		ACC[18] += RC * 0.24801366f;   

		ACC[22] += RC * -0.06886547f;   

		ACC[24] += RC * -0.15472333f;   

		ACC[38] += RC * -0.28598753f;   

		    RC = BC[156800 + C_offset + lane];
    
		ACC[9] += RC * -0.41365308f;   

		ACC[11] += RC * 0.16044457f;   

		ACC[17] += RC * 0.28503844f;   

		ACC[21] += RC * 0.1494196f;   

		ACC[23] += RC * 0.32455277f;   

		ACC[31] += RC * 0.113498814f;   

		ACC[47] += RC * -0.17516118f;   

		ACC[57] += RC * 0.12623154f;   

		    RC = BC[159936 + C_offset + lane];
    
		ACC[1] += RC * 0.65042615f;   

		ACC[6] += RC * 0.42594796f;   

		ACC[18] += RC * -0.13597205f;   

		    RC = BC[163072 + C_offset + lane];
    
		ACC[2] += RC * -0.42983758f;   

		ACC[12] += RC * -0.21008573f;   

		ACC[21] += RC * -0.054850865f;   

		ACC[24] += RC * -0.18604255f;   

		ACC[25] += RC * -0.30103827f;   

		ACC[37] += RC * 0.34806022f;   

		ACC[44] += RC * -0.15596001f;   

		ACC[52] += RC * -0.21316518f;   

		ACC[55] += RC * 0.37439975f;   

		    RC = BC[166208 + C_offset + lane];
    
		ACC[46] += RC * 0.42773876f;   

		    RC = BC[172480 + C_offset + lane];
    
		ACC[0] += RC * -0.07687451f;   

		ACC[3] += RC * -0.53101f;   

		ACC[5] += RC * -0.44122908f;   

		ACC[11] += RC * 0.21450573f;   

		ACC[24] += RC * -0.24097344f;   

		ACC[27] += RC * -0.17830954f;   

		ACC[44] += RC * -0.6377197f;   

		ACC[53] += RC * 0.055405397f;   

		    RC = BC[175616 + C_offset + lane];
    
		ACC[9] += RC * -0.22386803f;   

		ACC[14] += RC * 0.24518697f;   

		ACC[21] += RC * 0.06983554f;   

		ACC[48] += RC * 0.19773705f;   

		ACC[50] += RC * 0.5686969f;   

		ACC[54] += RC * 0.117811844f;   

		ACC[59] += RC * 0.25699562f;   

		    RC = BC[178752 + C_offset + lane];
    
		ACC[34] += RC * 0.40574792f;   

		ACC[38] += RC * -0.21245319f;   

		ACC[41] += RC * -0.581772f;   

		ACC[59] += RC * 0.08565152f;   

		    RC = BC[181888 + C_offset + lane];
    
		ACC[8] += RC * 0.7309588f;   

		ACC[22] += RC * -0.09992223f;   

		ACC[24] += RC * 0.13385744f;   

		ACC[28] += RC * -0.36533684f;   

		ACC[39] += RC * -0.25117013f;   

		ACC[48] += RC * 0.25177416f;   

		ACC[49] += RC * -0.28545076f;   

		ACC[53] += RC * -0.06506008f;   

		ACC[61] += RC * -0.28616887f;   

		    RC = BC[185024 + C_offset + lane];
    
		ACC[38] += RC * -0.21065329f;   

		ACC[40] += RC * 0.49672085f;   

		    RC = BC[188160 + C_offset + lane];
    
		ACC[10] += RC * 0.48832548f;   

		ACC[13] += RC * 0.102604404f;   

		ACC[14] += RC * -0.2498683f;   

		ACC[16] += RC * -0.26095983f;   

		ACC[24] += RC * 0.17783828f;   

		ACC[39] += RC * -0.26379153f;   

		ACC[47] += RC * -0.11098959f;   

		ACC[48] += RC * 0.36261433f;   

		ACC[51] += RC * 0.7271661f;   

		    RC = BC[191296 + C_offset + lane];
    
		ACC[0] += RC * -0.0407001f;   

		ACC[5] += RC * 0.18842979f;   

		ACC[11] += RC * -0.19668639f;   

		ACC[13] += RC * 0.22303726f;   

		ACC[19] += RC * -0.48006082f;   

		ACC[24] += RC * 0.14501482f;   

		ACC[27] += RC * -0.39214194f;   

		ACC[40] += RC * -0.34364948f;   

		ACC[53] += RC * 0.07577099f;   

		ACC[63] += RC * -0.27511463f;   

		    RC = BC[194432 + C_offset + lane];
    
		ACC[0] += RC * 0.08058559f;   

		ACC[5] += RC * -0.51369154f;   

		ACC[11] += RC * 0.29554477f;   

		ACC[16] += RC * -0.14971066f;   

		ACC[17] += RC * -0.2622464f;   

		ACC[18] += RC * 0.31855294f;   

		ACC[19] += RC * 1.2430506f;   

		ACC[22] += RC * -0.06381879f;   

		ACC[24] += RC * -0.4359843f;   

		ACC[27] += RC * 0.56794626f;   

		ACC[38] += RC * -0.17689636f;   

		ACC[44] += RC * -0.3349003f;   

		ACC[48] += RC * 0.3092569f;   

		ACC[53] += RC * -0.07257747f;   

		ACC[63] += RC * 0.22388396f;   

		    RC = BC[197568 + C_offset + lane];
    
		ACC[5] += RC * 0.17898779f;   

		ACC[18] += RC * -0.19749318f;   

		ACC[22] += RC * -0.051323816f;   

		ACC[24] += RC * 0.22011934f;   

		ACC[38] += RC * -0.3539517f;   

		ACC[54] += RC * -0.10591654f;   

	}



        AC[200704 + C_offset  + lane] = max(ACC[0] + 3.162181f,0.0f);

        AC[203840 + C_offset  + lane] = max(ACC[1] + -2.5823455f,0.0f);

        AC[206976 + C_offset  + lane] = max(ACC[2] + -2.7919135f,0.0f);

        AC[210112 + C_offset  + lane] = max(ACC[3] + 1.7371584f,0.0f);

        AC[213248 + C_offset  + lane] = max(ACC[4] + 3.1848283f,0.0f);

        AC[216384 + C_offset  + lane] = max(ACC[5] + -0.24944139f,0.0f);

        AC[219520 + C_offset  + lane] = max(ACC[6] + 0.014434099f,0.0f);

        AC[222656 + C_offset  + lane] = max(ACC[7] + -2.940038f,0.0f);

        AC[225792 + C_offset  + lane] = max(ACC[8] + -9.4151325f,0.0f);

        AC[228928 + C_offset  + lane] = max(ACC[9] + -3.1973352f,0.0f);

        AC[232064 + C_offset  + lane] = max(ACC[10] + -5.6700807f,0.0f);

        AC[235200 + C_offset  + lane] = max(ACC[11] + -6.306308f,0.0f);

        AC[238336 + C_offset  + lane] = max(ACC[12] + 1.8398961f,0.0f);

        AC[241472 + C_offset  + lane] = max(ACC[13] + 1.7022657f,0.0f);

        AC[244608 + C_offset  + lane] = max(ACC[14] + 5.3732634f,0.0f);

        AC[247744 + C_offset  + lane] = max(ACC[15] + 4.751031f,0.0f);

        AC[250880 + C_offset  + lane] = max(ACC[16] + 11.078623f,0.0f);

        AC[254016 + C_offset  + lane] = max(ACC[17] + -2.0852878f,0.0f);

        AC[257152 + C_offset  + lane] = max(ACC[18] + 8.101216f,0.0f);

        AC[260288 + C_offset  + lane] = max(ACC[19] + 6.377058f,0.0f);

        AC[263424 + C_offset  + lane] = max(ACC[20] + 2.3062682f,0.0f);

        AC[266560 + C_offset  + lane] = max(ACC[21] + 2.4351485f,0.0f);

        AC[269696 + C_offset  + lane] = max(ACC[22] + 5.1611986f,0.0f);

        AC[272832 + C_offset  + lane] = max(ACC[23] + 3.1691751f,0.0f);

        AC[275968 + C_offset  + lane] = max(ACC[24] + 0.50136256f,0.0f);

        AC[279104 + C_offset  + lane] = max(ACC[25] + -0.3714794f,0.0f);

        AC[282240 + C_offset  + lane] = max(ACC[26] + 3.2185884f,0.0f);

        AC[285376 + C_offset  + lane] = max(ACC[27] + 15.205744f,0.0f);

        AC[288512 + C_offset  + lane] = max(ACC[28] + -2.0926886f,0.0f);

        AC[291648 + C_offset  + lane] = max(ACC[29] + 3.2758942f,0.0f);

        AC[294784 + C_offset  + lane] = max(ACC[30] + -1.7504005f,0.0f);

        AC[297920 + C_offset  + lane] = max(ACC[31] + 5.1391497f,0.0f);

        AC[301056 + C_offset  + lane] = max(ACC[32] + 3.60483f,0.0f);

        AC[304192 + C_offset  + lane] = max(ACC[33] + -5.7226353f,0.0f);

        AC[307328 + C_offset  + lane] = max(ACC[34] + 2.104572f,0.0f);

        AC[310464 + C_offset  + lane] = max(ACC[35] + 2.8668141f,0.0f);

        AC[313600 + C_offset  + lane] = max(ACC[36] + -0.16970116f,0.0f);

        AC[316736 + C_offset  + lane] = max(ACC[37] + -0.23280361f,0.0f);

        AC[319872 + C_offset  + lane] = max(ACC[38] + 11.146543f,0.0f);

        AC[323008 + C_offset  + lane] = max(ACC[39] + 9.825462f,0.0f);

        AC[326144 + C_offset  + lane] = max(ACC[40] + 2.9263065f,0.0f);

        AC[329280 + C_offset  + lane] = max(ACC[41] + 2.70706f,0.0f);

        AC[332416 + C_offset  + lane] = max(ACC[42] + 10.834305f,0.0f);

        AC[335552 + C_offset  + lane] = max(ACC[43] + 2.3022428f,0.0f);

        AC[338688 + C_offset  + lane] = max(ACC[44] + -1.3121259f,0.0f);

        AC[341824 + C_offset  + lane] = max(ACC[45] + 1.028136f,0.0f);

        AC[344960 + C_offset  + lane] = max(ACC[46] + 4.571438f,0.0f);

        AC[348096 + C_offset  + lane] = max(ACC[47] + 6.4234734f,0.0f);

        AC[351232 + C_offset  + lane] = max(ACC[48] + -14.035306f,0.0f);

        AC[354368 + C_offset  + lane] = max(ACC[49] + 1.1635358f,0.0f);

        AC[357504 + C_offset  + lane] = max(ACC[50] + 0.11034632f,0.0f);

        AC[360640 + C_offset  + lane] = max(ACC[51] + -16.26934f,0.0f);

        AC[363776 + C_offset  + lane] = max(ACC[52] + 1.2764599f,0.0f);

        AC[366912 + C_offset  + lane] = max(ACC[53] + 5.10857f,0.0f);

        AC[370048 + C_offset  + lane] = max(ACC[54] + 2.3454251f,0.0f);

        AC[373184 + C_offset  + lane] = max(ACC[55] + 1.9230678f,0.0f);

        AC[376320 + C_offset  + lane] = max(ACC[56] + 0.96652675f,0.0f);

        AC[379456 + C_offset  + lane] = max(ACC[57] + 1.0347203f,0.0f);

        AC[382592 + C_offset  + lane] = max(ACC[58] + 1.883526f,0.0f);

        AC[385728 + C_offset  + lane] = max(ACC[59] + 0.81246436f,0.0f);

        AC[388864 + C_offset  + lane] = max(ACC[60] + 1.2826749f,0.0f);

        AC[392000 + C_offset  + lane] = max(ACC[61] + 3.50684f,0.0f);

        AC[395136 + C_offset  + lane] = max(ACC[62] + 4.6486897f,0.0f);

        AC[398272 + C_offset  + lane] = max(ACC[63] + -5.8354063f,0.0f);

}

 
}
int main()
{

	std::cout << "Group size " << Gsy << std::endl;

	cnpy::NpyArray arr = cnpy::npy_load("mobilenet/contraction_1x1_1_transposed.npy");
	float * AB = arr.data<float>();
	assert(arr.word_size = sizeof(float));
	assert(arr.shape.size()==2 && arr.shape[0] == 64 && arr.shape[1] == 128); //transposed

	cnpy::NpyArray arr1 = cnpy::npy_load("BC.npy");
	float * BC = arr1.data<float>();
	assert(arr1.word_size = sizeof(float));
#if In_Format == 'NHWC'
	assert(arr1.shape.size()==2 && arr1.shape[0] == 3136 && arr1.shape[1] == 64);
#else
	assert(arr1.shape.size()==2 && arr1.shape[0] == 64 && arr1.shape[1] == 3136);
#endif
    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
	float * AC = arr2.data<float>();
    std::cout << AC[0] << std::endl;

	float *d_BC, *d_AC;
	cudaMalloc((void**)&d_BC, 64 * 3136 *sizeof(float));
	cudaMalloc((void**)&d_AC, 128 * 3136 *sizeof(float));


	cudaMemcpy( d_BC,BC, 64 * 3136 *sizeof(float), cudaMemcpyHostToDevice);

	float *result;
	result = (float *)malloc(128 * 3136 *sizeof(result));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 GS(2,98);

     std::cout << "warning: sometimes you might want to fix the launch dimensions to 32" << std::endl;

    for(int i = 0;i < 1000;i ++){
	    mm<<<GS,Gsy>>>(d_BC,d_AC);
    }

	cudaProfilerStart();
	cudaEventRecord(start);

	for(int i = 0;i < 1000;i ++){
	    mm<<<GS,Gsy>>>(d_BC,d_AC);
    }
	cudaEventRecord(stop);
	cudaProfilerStop();
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "kernel used " << time / 1000.0 << std::endl;

	cudaMemcpy(result, d_AC, 128 * 3136 *sizeof(float), cudaMemcpyDeviceToHost);

	float error = 0;
	for(int i = 0 ; i < 128 * 3136; i ++)
	{
        error += abs(result[i] - AC[i]);
	}
	
	#if Out_Format == 'NCHW'
        cnpy::npy_save("result.npy",&result[0],{128,3136},"w");
    #else
        cnpy::npy_save("result.npy",&result[0],{3136,128},"w");
    #endif

	std::cout << result[0] << result[1] << result[2] << std::endl;
	std::cout << error << std::endl;
	cudaFree(d_BC);
	cudaFree(d_AC);
}
