# shareCode
## Explanation to the parameters of the function
### Input
* `K`: The observation range. <br>
* `trainInput`: The input signal.<br>
* `trainTarget`: The target signal.<br>
* `paramRegularization`: The regularization factor.<br>
* `typeKernel`: The choice of the kernel function.<br>
* `paramKernel`: The kernel width.<br>
* `stepSize`: The step-size parameter.<br>

### Output
`expansionCoefficient`: The expansion coefficient vector.<br>
`dictionaryIndex`: the index of the dictionary.<br>
`learningCurve`: The MSE convergence curve.<br>
`netSizeDiagram`: The record of increasing dictionary size.<br>
