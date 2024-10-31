const Diagnose_Patient = () => {
    const symptoms = document.getElementById('symptoms_input').value;
    const lab_test = document.getElementById('lab_test_input').value;
    const predicted_disease_input = document.getElementById('predicted_disease')

    const combined_text = `${symptoms} ${lab_test}`
    console.log(combined_text)
    axios.post('http://127.0.0.1:8080/predict',
        { content: combined_text },
        {
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log(response)
            predicted_disease_input.textContent = response.data.prediction
        })
}