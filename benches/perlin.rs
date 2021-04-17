use colored_noise::PerlinNoise;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use ndarray::prelude::*;

fn perlin2d_bench(c: &mut Criterion) {
    c.bench_function("perlin noise 2d", |b| {
        b.iter(|| {
            let x = Array1::<f64>::linspace(0., 1., 200);
            let y = Array1::<f64>::linspace(0., 1., 200);
            let _v: Array1<_> = x
                .iter()
                .cartesian_product(y.iter())
                .map(|(&x, &y)| PerlinNoise::new(200, 8).gen(&aview1(&[x, y])))
                .collect();
        })
    });
}

criterion_group!(benches, perlin2d_bench);
criterion_main!(benches);
