#[cfg(test)]
mod tests {
    use std::vec;
    use weldnet::{
        activation::{
            linear::ActFnLinear, relu::ActFnRelu, sigmoid::ActFnSigmoid, softmax::ActFnSoftmax,
            step::ActFnStep,
        },
        matrix::Matrix,
    };

    #[test]
    fn test_linear() {
        let input = Matrix::new(2, 3, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let mut activation = ActFnLinear::new();
        activation.forward(&input);
        let output = activation.get_output().as_ref().unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 6);
        assert_eq!(output.data, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_relu() {
        let input = Matrix::new(2, 3, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let mut activation = ActFnRelu::new();
        activation.forward(&input);
        let output = activation.get_output().as_ref().unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 6);
        assert_eq!(output.data, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sigmoid() {
        let input = Matrix::new(2, 3, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let mut activation = ActFnSigmoid::new();
        activation.forward(&input);
        let output = activation.get_output().as_ref().unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 6);
        assert_eq!(
            output.data,
            vec![
                0.11920292202211755,
                0.2689414213699951,
                0.5,
                0.7310585786300049,
                0.8807970779778824,
                0.9525741268224333,
            ]
        );
    }

    #[test]
    fn test_softmax() {
        let input = Matrix::new(2, 3, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let mut activation = ActFnSoftmax::new();
        activation.forward(&input);
        let output = activation.get_output().as_ref().unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 6);
        assert_eq!(
            output.data,
            vec![
                0.09003057317038046,
                0.24472847105479764,
                0.6652409557748218,
                0.09003057317038046,
                0.24472847105479764,
                0.6652409557748218,
            ]
        );
    }

    #[test]
    fn test_step() {
        let input = Matrix::new(2, 3, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let mut activation = ActFnStep::new();
        activation.forward(&input);
        let output = activation.get_output().as_ref().unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 6);
        assert_eq!(output.data, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }
}
