use crate::matrix::Matrix;

pub struct ActFnStep {
    inputs: Option<Matrix>,
    output: Option<Matrix>,
    // gradients
    d_inputs: Option<Matrix>,
}

impl ActFnStep {
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
            for j in 0..output.cols {
                if output.get(i, j) <= 0.0 {
                    output.set(i, j, 0.0);
                } else {
                    output.set(i, j, 1.0);
                }
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
