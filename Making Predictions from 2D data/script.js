console.log('Hello from TF')

async function getData(){
  const req = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
  const data = await req.json()
  const cleaned = data.map(obj => ({
    mpg: obj.Miles_per_Gallon,
    hp: obj.Horsepower
  })).filter(obj => (obj.mpg != null && obj.hp != null))
  
  return cleaned
}

function createModel(){
  const model = tf.sequential()

  // Hidden layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))

  // Another hidden layer
  // model.add(tf.layers.dense({inputShape: [1], units: 50, activation: 'sigmoid', useBias: true}))


 // model.add(tf.layers.dense({units: 10, activation: 'sigmoid'}))
  

  // Output layer
  model.add(tf.layers.dense({units: 1, useBias: true}))
  return model
}

function convertToTensor(data){

  // Clean up memory
  return tf.tidy(() => {
    tf.util.shuffle(data)
    
    const inputs = data.map(d => d.hp)
    const labels = data.map(d => d.mpg)

    const inputTensors = tf.tensor2d(inputs, [inputs.length, 1])  
    const labelTensors = tf.tensor2d(labels, [labels.length, 1])

    const [inputMax, inputMin] = [inputTensors.max(), inputTensors.min()]   
    const [labelMax, labelMin] = [labelTensors.max(), labelTensors.min()]

    const normalisedInputs = inputTensors.sub(inputMin).div(inputMax.sub(inputMin))
    const normalisedLabels = labelTensors.sub(labelMin).div(labelMax.sub(labelMin))

    return{
      inputs: normalisedInputs,
      labels: normalisedLabels,
      inputMax, inputMin, labelMax, labelMin
    }

  })
}


async function trainModel(model, inputs, labels){
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  })

  const batchSize = 28, epochs = 50

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      {name: 'Training Summary'},
      ['loss', 'mse'],
      {height: 200, callbacks: ['onEpochEnd']}
    )
  })
}

function testModel(model, inputData, normalizedData){
  const {inputMax, inputMin, labelMin, labelMax} = normalizedData

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100) // Create an array of 0 - 100 with a interval of 1
    const preds = model.predict(xs.reshape([100, 1])) // Actual prediction
    
    // Un- normalize means converting the data back to its original range
    const unNormalizedXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin)
    
    const unNormalizedPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin)

    // dataSync is a function used to get the values from Tensors
    return [unNormalizedXs.dataSync(), unNormalizedPreds.dataSync()]
  })

  // Predicted by model
  const predicts = Array.from(xs).map((x, ind) => ({x, y: preds[ind]}))
  // From original data
  const originals = inputData.map(d => ({x: d.hp, y: d.mpg}))

  tfvis.render.scatterplot(
    {name: 'Prediction vs. Original'},
    {values: [originals, predicts], series: ['Original', 'Predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
}

async function run(){
  const data = await getData()
  const values = data.map(obj => ({
    x: obj.hp,
    y: obj.mpg
  }))

  tfvis.render.scatterplot(
    {name: 'Horsepower vs MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'Miles per Gallon',
      height: 300
    }
  )

  const model = createModel()
  tfvis.show.modelSummary({name: 'Model Summary'}, model)

  const dataInTensors = convertToTensor(data)
  const {inputs, labels} = dataInTensors

  await trainModel(model, inputs, labels)
  console.log('Training done')

  testModel(model, data, dataInTensors)
}




document.addEventListener('DOMContentLoaded', run)
