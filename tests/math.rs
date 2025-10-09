#[cfg(test)]
mod tests {
    use std::vec;
    use weldnet::math::Math;
    use weldnet::matrix::Matrix;

    #[test]
    fn test_transpose() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = Math::transpose(&matrix);
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.data.len(), 6);
        assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_product() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let product = Math::product(&a, &b);
        assert_eq!(product.rows, 2);
        assert_eq!(product.cols, 2);
        assert_eq!(product.data.len(), 4);
        assert_eq!(product.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    #[should_panic(expected = "Matrix size doesn't match")]
    fn test_product_invalid() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        Math::product(&a, &b);
    }

    #[test]
    fn test_add_row() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = Matrix::new(1, 3, vec![7.0, 8.0, 9.0]);
        let result = Math::add_row(&matrix, &row);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert_eq!(result.data.len(), 6);
        assert_eq!(result.data, vec![8.0, 10.0, 12.0, 11.0, 13.0, 15.0]);
    }

    #[test]
    #[should_panic(expected = "Expected a single row matrix")]
    fn test_add_row_invalid_row() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = Matrix::new(3, 1, vec![7.0, 8.0, 9.0]);
        Math::add_row(&matrix, &row);
    }

    #[test]
    #[should_panic(expected = "Matrix columns doesn't match single row matrix columns")]
    fn test_add_row_invalid_col() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = Matrix::new(1, 2, vec![7.0, 8.0]);
        Math::add_row(&matrix, &row);
    }

    #[test]
    fn test_reduce_sum() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = Math::reduce_sum(&matrix);
        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, 3);
        assert_eq!(result.data.len(), 3);
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }
}
