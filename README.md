# fluid-animations

My codes for [The Art of Fluid Animation](https://www.routledge.com/The-Art-of-Fluid-Animation/Stam/p/book/9781498700207).

## Run

```
cargo run --release --bin 01_diffuse
```

Open images in `out` directory.

# Demos

![01_diffuse](demos/01_diffuse.mp4)

## Cheat sheet

Create movie file from pngs

```
ffmpeg -r 24 -i out/%06d.png -pix_fmt yuv420p out.mp4
```