bins=("01_diffuse" "02_dens_step" "03_vector_noise" "04_multiple_densities" "05_vel_step")

if [ $# -ne 0 ]; then
	bins=(${@})
fi

for bin in "${bins[@]}"; do
	echo $bin
	rm out/*
	cargo run --release --bin ${bin}
	ffmpeg -r 24 -i out/%06d.png -pix_fmt yuv420p demos/"${bin}".gif -y
done