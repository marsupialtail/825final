
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

__global__ void mm(const float * __restrict__ BC, const float * __restrict__ BA, const float * __restrict__ bias, float * AC)
{
    register float ACC[128] = {0.0};
	register float RC = 0.0;
#if Gy > 1	
        __shared__ float result[128][Tsz];
	for(int i = threadIdx.x; i < 128 * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();
#endif
#if In_Format == 'NHWC'
	__shared__ float smem_cache[Tsz][32+1];
#endif
#if Out_Format == 'NHWC'
	__shared__ float smem_result[Tsz][128+1];
#endif

	int A_offset = blockIdx.x * (128 / 1);
	int C_offset = blockIdx.y * (3136 / 98);
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);


if(blockIdx.x == 0)
{


int A_offset = 0;
int block_NY = 128;

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


	if(groupId == 1)
	{

		for (int b_idx = 0; b_idx < 64; b_idx++)
		{
		for (int ny_idx = block_NY/2; ny_idx < block_NY; ny_idx++)
		{
		int a_idx = 0 + ny_idx;
		RC = BC[b_idx * 3136 + lane];
		ACC[ny_idx] += RC * BA[b_idx, a_idx];
		}
		}

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
for (int i = block_NY/2; i < block_NY; i++)
{

        AC[(A_offset + i) * 3136 + C_offset  + lane] = max(ACC[i] + bias[A_offset+i],0.0f);
}
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
	
	cnpy::NpyArray arr4 = cnpy::npy_load("mobilenet/contraction_1x1_1_bias.npy");
	float * bias = arr4.data<float>();
	assert(arr4.word_size = sizeof(float));
#if In_Format == 'NHWC'
	assert(arr4.shape.size()==1 && arr4.shape[0] == 128);
#else
	assert(arr4.shape.size()==1 && arr4.shape[0] == 128);
#endif
	
    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
	float * AC = arr2.data<float>();
    std::cout << AC[0] << std::endl;

	float *d_BC, *d_AC;
	float *d_BA, *d_bias;
	cudaMalloc((void**)&d_BC, 64 * 3136 *sizeof(float));
	cudaMalloc((void**)&d_AC, 128 * 3136 *sizeof(float));
	cudaMalloc((void**)&d_BA, 64 * 128 *sizeof(float));
	cudaMalloc((void**)&d_bias, 64 * 3136 *sizeof(float));


	cudaMemcpy( d_BC,BC, 64 * 3136 *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( d_bias,bias, 64 * 3136 *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( d_BA,AB, 64 * 128 *sizeof(float), cudaMemcpyHostToDevice);

	float *result;
	result = (float *)malloc(128 * 3136 *sizeof(result));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 GS(1,98);

     std::cout << "warning: sometimes you might want to fix the launch dimensions to 32" << std::endl;
    // We now launch twice the number of threads
    for(int i = 0;i < 1000;i ++){
	    mm<<<GS,Gsy * 2>>>(d_BC,d_AC,d_BA,d_bias);
    }

	cudaProfilerStart();
	cudaEventRecord(start);

	for(int i = 0;i < 1000;i ++){
	    mm<<<GS,Gsy * 2>>>(d_BC,d_AC,d_BA,d_bias);
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
