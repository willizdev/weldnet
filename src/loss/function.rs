use crate::matrix::Matrix;

pub struct LossFn;

impl LossFn {
    pub fn from_onehot(input: &Matrix) -> Vec<usize> {
        let mut res: Vec<usize> = Vec::new();
        for i in 0..input.rows {
            for j in 0..input.cols {
                if input.get(i, j) == 1.0 {
                    res.push(j);
                    break;
                }
            }
        }
        res
    }

    pub fn calculate(input: &Matrix, target: &Vec<usize>) -> f64 {
        let epsilon: f64 = 1e-15;
        let mut losses: f64 = 0.0;

        for i in 0..input.rows {
            let j: usize = target[i];
            let pred: f64 = input.get(i, j).clamp(epsilon, 1.0 - epsilon);
            losses -= pred.ln();
        }

        losses / input.rows as f64
    }

    pub fn accuracy(input: &Matrix, target: &Vec<usize>) -> f64 {
        let mut correct: f64 = 0.0;

        for i in 0..input.rows {
            let mut k: usize = 0;
            let mut m: f64 = input.get(i, k);
            for j in 1..input.cols {
                let n: f64 = input.get(i, j);
                if n > m {
                    m = n;
                    k = j;
                }
            }
            if target[i] == k {
                correct += 1.0;
            }
        }

        correct / target.len() as f64
    }
}
