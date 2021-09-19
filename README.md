# fluid-animations

My codes for [The Art of Fluid Animation](https://www.routledge.com/The-Art-of-Fluid-Animation/Stam/p/book/9781498700207) (01 to 05) and [Fluid Simulation for Computer Graphics](https://www.routledge.com/Fluid-Simulation-for-Computer-Graphics/Bridson/p/book/9781482232837) (06-).

## Run

```
cargo run --release --bin 01_diffuse
```

Open images in `out` directory.

# Demos

## 01_diffuse

![01_diffuse](demos/01_diffuse.gif)

## 02_dens_step

![02_dens_step](demos/02_dens_step.gif)

## 03_vector_noise

![03_vector_noise](demos/03_vector_noise.gif)

## 04_multiple_densities

![04_multiple_densities](demos/04_multiple_densities.gif)

## 05_vel_step

![05_vel_step](demos/05_vel_step.gif)

## 06_advect_improved

![06_advect_improved](demos/06_advect_improved.gif)
## Cheat sheet

Create movie file from pngs

```
ffmpeg -r 24 -i out/%06d.png -pix_fmt yuv420p out.mp4
```