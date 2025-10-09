use crate::matrix::Matrix;

pub struct ActFnSoftmax {
    inputs: Option<Matrix>,
    output: Option<Matrix>,
    // gradients
    d_inputs: Option<Matrix>,
}

impl ActFnSoftmax {
    pub fn new() -> Self {
        Self {
            inputs: None,
            output: None,
            d_inputs: None,
        }
    }

    pub fn forward(&mut self, inputs: &Matrix) {
        let mut output: Matrix = inputs.clone();

        for i in 0..output.rows {
            let mut max: f64 = output.get(i, 0);
            for j in 1..output.cols {
                max = max.max(output.get(i, j));
            }
            let mut sum: f64 = 0.0;
            for j in 0..output.cols {
                // we subtract by the max to prevent overflow
                let exp: f64 = (output.get(i, j) - max).exp();
                output.set(i, j, exp);
                sum += exp;
            }
            for j in 0..output.cols {
                let norm: f64 = output.get(i, j) / sum;
                output.set(i, j, norm);
            }
        }

        self.output = Some(output);
        // input is stored for backpropagation
        self.inputs = Some(inputs.clone());
    }

    // TODO: implement the derivative
    #[allow(dead_code)]
    fn backward(&mut self, d_values: &Matrix) {
        // let inputs: &Matrix = self.inputs.as_ref().unwrap();
        let d_inputs: Matrix = d_values.clone();
        self.d_inputs = Some(d_inputs);
    }

    pub fn get_output(&self) -> &Option<Matrix> {
        &self.output
    }
}
