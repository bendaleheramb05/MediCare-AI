/* MediCare AI - client-side helpers */

// Live BMI calculation on the assessment form
document.addEventListener("DOMContentLoaded", function () {
  const h = document.getElementById("height");
  const w = document.getElementById("weight");
  const out = document.getElementById("bmi");
  if (!h || !w || !out) return;

  function updateBMI() {
    const height = parseFloat(h.value), weight = parseFloat(w.value);
    if (height > 0 && weight > 0) {
      const bmi = weight / Math.pow(height / 100, 2);
      let label = "Obese";
      if (bmi < 18.5) label = "Underweight";
      else if (bmi < 25) label = "Normal";
      else if (bmi < 30) label = "Overweight";
      out.value = bmi.toFixed(1) + "  (" + label + ")";
    } else {
      out.value = "";
    }
  }
  h.addEventListener("input", updateBMI);
  w.addEventListener("input", updateBMI);
  updateBMI();

  const bpUnknown = document.getElementById("bpUnknown");
  const systolic = document.querySelector("[name='systolic_bp']");
  const diastolic = document.querySelector("[name='diastolic_bp']");
  if (bpUnknown && systolic && diastolic) {
    function updateBloodPressureInputs() {
      const disabled = bpUnknown.checked;
      systolic.disabled = disabled;
      diastolic.disabled = disabled;
      if (disabled) {
        systolic.value = "";
        diastolic.value = "";
      }
    }
    bpUnknown.addEventListener("change", updateBloodPressureInputs);
    updateBloodPressureInputs();
  }
});

// Simple sanity check before submitting the assessment form
const form = document.getElementById("assessForm");
if (form) {
  form.addEventListener("submit", function (e) {
    const sys = parseInt(form.systolic_bp.value, 10);
    const dia = parseInt(form.diastolic_bp.value, 10);
    if (sys && dia && dia >= sys) {
      e.preventDefault();
      alert("Diastolic (lower) blood pressure must be less than systolic (upper).");
    }
  });
}
