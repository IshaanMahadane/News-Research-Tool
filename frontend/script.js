// Replace this with your actual AWS Elastic Beanstalk backend URL after deployment
const BACKEND_URL = "https://your-app-name.eu-north-1.elasticbeanstalk.com";

async function processURLs() {
  const urls = [
    document.getElementById("url1").value,
    document.getElementById("url2").value,
    document.getElementById("url3").value,
  ].filter(u => u.trim() !== "");

  const status = document.getElementById("process-status");
  status.innerText = "⏳ Processing URLs...";

  try {
    const res = await fetch(`${BACKEND_URL}/process-urls`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
    });

    const data = await res.json();
    if (res.ok) {
      status.innerText = `✅ ${data.message}`;
    } else {
      status.innerText = `❌ Error: ${data.detail || data.message}`;
    }
  } catch (err) {
    status.innerText = "❌ Server connection failed.";
    console.error(err);
  }
}

async function askQuestion() {
  const question = document.getElementById("question").value.trim();
  const answerBox = document.getElementById("answer-box");
  const answer = document.getElementById("answer");

  if (!question) {
    answer.innerText = "⚠️ Please enter a question.";
    answerBox.style.display = "block";
    return;
  }

  answer.innerText = "💭 Thinking...";
  answerBox.style.display = "block";

  try {
    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    if (res.ok) {
      answer.innerText = data.answer || "No answer found.";
    } else {
      answer.innerText = `❌ Error: ${data.detail || "Something went wrong."}`;
    }
  } catch (err) {
    answer.innerText = "❌ Server not reachable.";
    console.error(err);
  }
}
