bins=("01_diffuse" "02_dens_step" "03_vector_noise" "04_multiple_densities" "05_vel_step")

for bin in "${bins[@]}"; do
	rm out/*
	cargo run --release --bin ${bin}
	ffmpeg -r 24 -i out/%06d.png -pix_fmt yuv420p demos/"${bin}".gif -y
done