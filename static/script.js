document
  .getElementById("predict-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = {
      pclass: document.getElementById("pclass").value,
      sex: document.getElementById("sex").value,
      age: document.getElementById("age").value,
      sibsp: document.getElementById("sibsp").value,
      parch: document.getElementById("parch").value,
      fare: document.getElementById("fare").value,
      embarked: document.getElementById("embarked").value,
    };

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

    const data = await response.json();
    if (data.error) {
      document.getElementById("result").innerText = `Error: ${data.error}`;
    } else {
      document.getElementById("result").innerText = `Survived: ${
        data.survived
      } (Probability: ${data.survival_probability.toFixed(2)})`;
    }
  });
