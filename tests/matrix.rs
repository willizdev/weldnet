#[cfg(test)]
mod tests {
    use std::vec;
    use weldnet::matrix::Matrix;

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data.len(), 6);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Data size doesn't match dimensions")]
    fn test_matrix_creation_invalid() {
        Matrix::new(4, 4, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matrix_zeros() {
        let matrix = Matrix::zeros(4, 4);
        assert_eq!(matrix.rows, 4);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data.len(), 16);
        assert_eq!(matrix.data, vec![0.0; 16]);
    }

    #[test]
    fn test_matrix_random() {
        let matrix = Matrix::random(4, 4, -1.0, 1.0);
        assert_eq!(matrix.rows, 4);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data.len(), 16);
        for &value in &matrix.data {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_matrix_get_set() {
        let mut matrix = Matrix::new(4, 4, vec![0.0; 16]);
        matrix.set(2, 1, 2.71);
        assert_eq!(matrix.get(2, 1), 2.71);
        matrix.set(0, 3, 3.14);
        assert_eq!(matrix.get(0, 3), 3.14);
    }
}
