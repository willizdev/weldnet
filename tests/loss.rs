#[cfg(test)]
mod tests {
    use std::vec;
    use weldnet::{loss::function::LossFn, matrix::Matrix};

    #[test]
    fn test_from_onehot() {
        let input: Matrix = Matrix::new(2, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let target: Vec<usize> = LossFn::from_onehot(&input);
        assert_eq!(target, vec![1, 0]);
    }

    #[test]
    fn test_calculate() {
        let input: Matrix = Matrix::new(2, 3, vec![0.025, 0.95, 0.025, 0.025, 0.025, 0.95]);
        let target: Vec<usize> = vec![1, 2];
        let loss: f64 = LossFn::calculate(&input, &target);
        assert!(loss < 0.1);
    }

    #[test]
    fn test_accuracy() {
        let input: Matrix = Matrix::new(2, 3, vec![0.025, 0.95, 0.025, 0.025, 0.025, 0.95]);
        let target: Vec<usize> = vec![1, 2];
        let acc: f64 = LossFn::accuracy(&input, &target);
        assert_eq!(acc, 1.0);
    }
}
