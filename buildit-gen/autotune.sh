python make_BC.py ../mobilenet/contraction_1x1_1_transposed.npy 3136 NCHW NCHW ../mobilenet/contraction_1x1_1_bias.npy
for A_blocks in 1 2 4; do
	for C_blocks in 49 98; do
		for Gy_i in 0 1 2 4; do
			for Gy_d in 0 1 2 4; do
				if [[ $Gy_i == 0 && $Gy_d == 0 ]]; then
					continue;
				fi
				bash eval.sh $A_blocks $C_blocks $Gy_i $Gy_d
			done
		done
	done

done
