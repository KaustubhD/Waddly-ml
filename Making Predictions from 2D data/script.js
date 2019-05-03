console.log('Hello from TF')

let getData = async () => {
  const req = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
  const data = await req.json()
  const cleaned = data.map(obj => ({
    mpg: obj.Miles_per_Gallon,
    hp: obj.Horsepower
  })).filter(obj => obj.mpg && obj.hp)
  
  return cleaned
}

let createModel = () => {
  const model = tf.sequential()

  // Hidden layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))

  // Output layer
  model.add(tf.layers.dense({units: 1, useBias: true}))
  return model
}




let run = async () => {
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
      height: 400
    }
  )

  const model = createModel()
  tfvis.show.modelSummary({name: 'Model Summary'}, model)
}




document.addEventListener('DOMContentLoaded', run)
