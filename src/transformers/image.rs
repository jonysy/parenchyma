use image::DynamicImage;

use super::super::Transformer;

impl Transformer for DynamicImage {
    fn as_vector(&self) -> Vec<f32> {
        self.raw_pixels().iter().map(|&elem| elem as f32).collect()
    }
}