import React, { useState } from "react";
import axios from "axios";

export default function Chatbot() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [summary, setSummary] = useState("");
  const [ttsUrl, setTtsUrl] = useState("");

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:8000/chat", {
        query: query,
        summarize: true,
        tts: true,
      });
      setResponse(res.data.response);
      setSummary(res.data.summary || "");
      setTtsUrl(res.data.audio_url || "");
    } catch (err) {
      console.error(err);
    }
  };

  const playAudio = () => {
    if (ttsUrl) {
      const audio = new Audio(`http://localhost:8000${ttsUrl}`);
      audio.play();
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "600px", margin: "auto" }}>
      <h2>AI Chatbot</h2>
      <input
        type="text"
        placeholder="Ask a question..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ width: "100%", padding: "0.5rem" }}
      />
      <button onClick={handleSubmit} style={{ marginTop: "1rem", padding: "0.5rem 1rem" }}>
        Submit
      </button>

      {response && (
        <div style={{ marginTop: "1.5rem" }}>
          <h3>Response:</h3>
          <p>{response}</p>
        </div>
      )}

      {summary && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Summary:</h3>
          <p>{summary}</p>
        </div>
      )}

      {ttsUrl && (
        <div style={{ marginTop: "1rem" }}>
          <button onClick={playAudio}>🔊 Play Audio</button>
        </div>
      )}
    </div>
  );
}
