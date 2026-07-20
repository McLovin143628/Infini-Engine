import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import PanelWindowApp from "./panels/window/PanelWindowApp";
import { initTheme } from "./lib/theme";
import "./styles.css";

initTheme();

// One bundle, two roots: the main editor window, and the lean detached
// panel windows spawned at `#/panel/<panelId>` (ROADMAP P1.2.4).
const isPanelWindow = window.location.hash.startsWith("#/panel/");

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>{isPanelWindow ? <PanelWindowApp /> : <App />}</React.StrictMode>,
);
