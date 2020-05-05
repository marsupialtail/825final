python make_BC.py ../mobilenet/contraction_1x1_6_transposed.npy 196 NCHW NCHW ../mobilenet/contraction_1x1_6_bias.npy
#for A_blocks in 8 16 32; do
for A_blocks in 32; do
        #for C_blocks in 1 2 4 7; do
        for C_blocks in 1; do
                for Gy_i in 1 2 3 4 5 6 7; do
			for Gy_d in 1; do
				if [[ $Gy_i == 0 && $Gy_d == 0 ]]; then
					continue;
				fi
				bash eval6.sh $A_blocks $C_blocks $Gy_i $Gy_d &
			done
		done
	done

done
