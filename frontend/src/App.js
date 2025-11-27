import { useState, useEffect, useCallback } from "react";
import "@/App.css";
import axios from "axios";
import Editor from "@monaco-editor/react";
import {
  Loader2,
  Play,
  Download,
  Share2,
  Code2,
  Sparkles,
  AlertCircle,
  Info,
  X,
  Zap,
  Activity,
  ChevronDown,
  ChevronUp,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const EXAMPLE_CODE = `from tensorflow import keras
from tensorflow.keras import layers

# Build a Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])`;

function App() {
  const [code, setCode] = useState(EXAMPLE_CODE);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [activePanel, setActivePanel] = useState("both"); // 'both', 'code', 'visualization'
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);

  const analyzeCode = useCallback(async (codeToAnalyze) => {
    if (!codeToAnalyze || codeToAnalyze.trim().length === 0) {
      setAnalysis(null);
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(
        `${API}/analyze-code`,
        {
          code: codeToAnalyze,
        },
        {
          timeout: 30000, // 30 second timeout
        }
      );
      setAnalysis(response.data);

      if (response.data.success) {
        toast.success("Architecture analyzed with AI!");
        // Auto-collapse editor and expand visualizer after successful analysis
        setActivePanel("visualization");
        setHasAnalyzed(true);
      } else if (response.data.error) {
        toast.error(response.data.error);
      }
    } catch (error) {
      console.error("Analysis error:", error);

      let errorMessage = "Failed to analyze code";

      if (
        error.code === "ECONNREFUSED" ||
        error.message?.includes("Network Error")
      ) {
        errorMessage =
          "Cannot connect to backend. Make sure the server is running on port 8000.";
      } else if (
        error.code === "ETIMEDOUT" ||
        error.message?.includes("timeout")
      ) {
        errorMessage = "Request timed out. Please try again.";
      } else if (error.response?.status === 500) {
        errorMessage = error.response?.data?.detail || "Server error occurred";
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      }

      toast.error(errorMessage);
      setAnalysis({
        success: false,
        error: errorMessage,
        layers: [],
        connections: [],
        warnings: [],
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      analyzeCode(code);
    }, 1500);
    return () => clearTimeout(timer);
  }, [code, analyzeCode]);

  const handleDownload = () => {
    if (!analysis || !analysis.success) return;

    const dataStr = JSON.stringify(analysis, null, 2);
    const dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);
    const exportFileDefaultName = "neural-architecture.json";

    const linkElement = document.createElement("a");
    linkElement.setAttribute("href", dataUri);
    linkElement.setAttribute("download", exportFileDefaultName);
    linkElement.click();
    toast.success("Architecture downloaded!");
  };

  const handleShare = async () => {
    if (!analysis || !analysis.success) return;

    try {
      await navigator.clipboard.writeText(JSON.stringify(analysis, null, 2));
      toast.success("Copied to clipboard!");
    } catch (error) {
      toast.error("Failed to copy");
    }
  };

  const handleNodeClick = (layer) => {
    setSelectedNode(layer);
  };

  const closeModal = () => {
    setSelectedNode(null);
  };

  // Calculate neuron positions for a layer with dynamic sizing
  const calculateNeuronPositions = (
    layer,
    layerIndex,
    totalLayers,
    canvasWidth,
    canvasHeight
  ) => {
    const neurons = layer.neurons || 4;
    const maxNeuronsToShow = Math.min(neurons, 6); // Reduced to fit better
    const positions = [];

    // Dynamic spacing based on number of layers
    const padding = 80;
    const availableWidth = canvasWidth - padding * 2;
    const xPos =
      totalLayers > 1
        ? padding + (layerIndex / (totalLayers - 1)) * availableWidth
        : canvasWidth / 2;

    // Dynamic vertical spacing based on canvas height
    const verticalPadding = 60;
    const availableHeight = canvasHeight - verticalPadding * 2 - 100; // Reserve space for labels
    const spacing = Math.min(50, availableHeight / (maxNeuronsToShow + 1));
    const startY =
      verticalPadding +
      (availableHeight - (maxNeuronsToShow - 1) * spacing) / 2;

    for (let i = 0; i < maxNeuronsToShow; i++) {
      positions.push({
        x: xPos,
        y: startY + i * spacing,
        neuronIndex: i,
      });
    }

    return { positions, totalNeurons: neurons, showing: maxNeuronsToShow };
  };

  // Calculate dynamic canvas dimensions based on layers
  const getCanvasDimensions = () => {
    if (!analysis?.layers) return { width: 1000, height: 500 };

    const layerCount = analysis.layers.length;
    // Width: minimum 150px per layer, but at least 1000px
    const width = Math.max(1000, layerCount * 150);
    // Height based on max neurons in any layer
    const maxNeurons = Math.max(
      ...analysis.layers.map((l) => l.neurons || 4),
      4
    );
    const height = Math.max(500, Math.min(maxNeurons * 60, 800));

    return { width, height };
  };

  return (
    <div className="app-container" data-testid="neural-canvas-app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <Activity className="w-7 h-7" />
            </div>
            <div>
              <h1 className="app-title">Neural Canvas</h1>
              <p className="app-subtitle">
                <Sparkles className="inline w-3.5 h-3.5 mr-1" />
                AI-Powered Architecture Visualization
              </p>
            </div>
          </div>
          <div className="header-actions">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              disabled={!analysis?.success}
              data-testid="download-btn"
              className="action-btn"
            >
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleShare}
              disabled={!analysis?.success}
              data-testid="share-btn"
              className="action-btn"
            >
              <Share2 className="w-4 h-4 mr-2" />
              Share
            </Button>
            <Button
              size="sm"
              onClick={() => analyzeCode(code)}
              disabled={loading}
              data-testid="analyze-btn"
              className="analyze-btn"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              Analyze
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className={`main-content ${isFullscreen ? "fullscreen-mode" : ""}`}>
        <div className={`content-layout ${activePanel}`}>
          {/* Code Editor Panel */}
          <div
            className={`code-panel ${
              activePanel === "visualization" ? "collapsed" : ""
            } ${activePanel === "code" ? "expanded" : ""}`}
          >
            <div
              className="panel-header clickable"
              onClick={() => {
                if (activePanel === "code") {
                  setActivePanel("both");
                } else {
                  setActivePanel("code");
                }
              }}
            >
              <div className="panel-header-left">
                <Code2 className="w-5 h-5" />
                <h2 className="panel-title">Code Editor</h2>
                {analysis?.framework && (
                  <Badge variant="outline" className="ml-2 framework-badge">
                    {analysis.framework}
                  </Badge>
                )}
              </div>
              <div className="panel-header-right">
                {activePanel === "visualization" ? (
                  <ChevronDown className="w-5 h-5 opacity-60" />
                ) : (
                  <ChevronUp className="w-5 h-5 opacity-60" />
                )}
              </div>
            </div>
            {activePanel !== "visualization" && (
              <div className="editor-container" data-testid="code-editor">
                <Editor
                  height="100%"
                  defaultLanguage="python"
                  value={code}
                  onChange={(value) => setCode(value || "")}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 13,
                    lineNumbers: "on",
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    tabSize: 4,
                    padding: { top: 12 },
                    wordWrap: "on",
                  }}
                />
              </div>
            )}
          </div>

          {/* Visualization Panel */}
          <div
            className={`visualization-panel ${
              activePanel === "code" ? "collapsed" : ""
            } ${activePanel === "visualization" ? "expanded" : ""} ${
              isFullscreen ? "fullscreen" : ""
            }`}
          >
            <div
              className="panel-header clickable"
              onClick={() => {
                if (isFullscreen) return;
                if (activePanel === "visualization") {
                  setActivePanel("both");
                } else {
                  setActivePanel("visualization");
                }
              }}
            >
              <div className="panel-header-left">
                <Sparkles className="w-5 h-5" />
                <h2 className="panel-title">Neural Network</h2>
              </div>
              {loading && (
                <div className="ml-2 flex items-center text-sm analyzing-badge">
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </div>
              )}
              <div className="panel-header-right">
                {activePanel === "code" ? (
                  <ChevronDown className="w-5 h-5 opacity-60" />
                ) : (
                  <ChevronUp className="w-5 h-5 opacity-60" />
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsFullscreen(!isFullscreen);
                  }}
                  className="fullscreen-btn"
                >
                  {isFullscreen ? (
                    <Minimize2 className="w-4 h-4" />
                  ) : (
                    <Maximize2 className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </div>

            <div
              className="visualization-container"
              data-testid="visualization-canvas"
            >
              {analysis?.error && (
                <div className="error-state">
                  <AlertCircle className="w-12 h-12 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Analysis Error</h3>
                  <p className="text-sm opacity-80">{analysis.error}</p>
                </div>
              )}

              {analysis?.success &&
                analysis?.layers &&
                analysis.layers.length > 0 && (
                  <div className="visualization-content">
                    {/* Architecture Summary */}
                    {analysis.architecture_summary && (
                      <div className="architecture-summary">
                        <Info className="w-4 h-4" />
                        <span>{analysis.architecture_summary}</span>
                      </div>
                    )}

                    {/* Warnings */}
                    {analysis.warnings && analysis.warnings.length > 0 && (
                      <div className="warning-banner">
                        <AlertCircle className="w-4 h-4" />
                        <span>{analysis.warnings[0]}</span>
                      </div>
                    )}

                    {/* Neural Network Canvas */}
                    <div className="neural-canvas">
                      {(() => {
                        const { width: canvasWidth, height: canvasHeight } =
                          getCanvasDimensions();
                        return (
                          <svg
                            width={canvasWidth}
                            height={canvasHeight}
                            className="neural-svg"
                            viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
                          >
                            <defs>
                              <linearGradient
                                id="connectionGradient"
                                x1="0%"
                                y1="0%"
                                x2="100%"
                                y2="0%"
                              >
                                <stop
                                  offset="0%"
                                  style={{
                                    stopColor: "#6366f1",
                                    stopOpacity: 0.2,
                                  }}
                                />
                                <stop
                                  offset="100%"
                                  style={{
                                    stopColor: "#8b5cf6",
                                    stopOpacity: 0.2,
                                  }}
                                />
                              </linearGradient>
                              <filter id="glow">
                                <feGaussianBlur
                                  stdDeviation="2"
                                  result="coloredBlur"
                                />
                                <feMerge>
                                  <feMergeNode in="coloredBlur" />
                                  <feMergeNode in="SourceGraphic" />
                                </feMerge>
                              </filter>
                            </defs>

                            {/* Draw connections between neurons */}
                            {analysis.layers.map((layer, layerIdx) => {
                              if (layerIdx === analysis.layers.length - 1)
                                return null;

                              const currentLayer = calculateNeuronPositions(
                                layer,
                                layerIdx,
                                analysis.layers.length,
                                canvasWidth,
                                canvasHeight
                              );
                              const nextLayer = calculateNeuronPositions(
                                analysis.layers[layerIdx + 1],
                                layerIdx + 1,
                                analysis.layers.length,
                                canvasWidth,
                                canvasHeight
                              );

                              return currentLayer.positions.map(
                                (sourcePos, sourceIdx) =>
                                  nextLayer.positions.map(
                                    (targetPos, targetIdx) => (
                                      <line
                                        key={`conn-${layerIdx}-${sourceIdx}-${targetIdx}`}
                                        x1={sourcePos.x}
                                        y1={sourcePos.y}
                                        x2={targetPos.x}
                                        y2={targetPos.y}
                                        stroke="url(#connectionGradient)"
                                        strokeWidth="1"
                                        opacity="0.3"
                                        className="connection-line"
                                      />
                                    )
                                  )
                              );
                            })}

                            {/* Draw neurons */}
                            {analysis.layers.map((layer, layerIdx) => {
                              const neuronData = calculateNeuronPositions(
                                layer,
                                layerIdx,
                                analysis.layers.length,
                                canvasWidth,
                                canvasHeight
                              );

                              return (
                                <g key={`layer-${layerIdx}`}>
                                  {neuronData.positions.map(
                                    (pos, neuronIdx) => (
                                      <circle
                                        key={`neuron-${layerIdx}-${neuronIdx}`}
                                        cx={pos.x}
                                        cy={pos.y}
                                        r="16"
                                        className={`neuron ${
                                          hoveredNode?.id === layer.id
                                            ? "hovered"
                                            : ""
                                        } ${
                                          selectedNode?.id === layer.id
                                            ? "selected"
                                            : ""
                                        }`}
                                        onClick={() => handleNodeClick(layer)}
                                        onMouseEnter={() =>
                                          setHoveredNode(layer)
                                        }
                                        onMouseLeave={() =>
                                          setHoveredNode(null)
                                        }
                                        data-testid={`neuron-${layerIdx}-${neuronIdx}`}
                                        filter={
                                          selectedNode?.id === layer.id ||
                                          hoveredNode?.id === layer.id
                                            ? "url(#glow)"
                                            : ""
                                        }
                                      />
                                    )
                                  )}

                                  {/* Layer label - truncated and rotated for better readability */}
                                  <g
                                    transform={`translate(${
                                      neuronData.positions[0].x
                                    }, ${
                                      neuronData.positions[
                                        neuronData.positions.length - 1
                                      ].y + 40
                                    })`}
                                  >
                                    <text
                                      x="0"
                                      y="0"
                                      textAnchor="middle"
                                      className="layer-label"
                                      onClick={() => handleNodeClick(layer)}
                                    >
                                      {layer.name?.length > 12
                                        ? layer.name.substring(0, 12) + "..."
                                        : layer.name}
                                    </text>

                                    {/* Layer type badge */}
                                    <text
                                      x="0"
                                      y="16"
                                      textAnchor="middle"
                                      className="layer-type"
                                      onClick={() => handleNodeClick(layer)}
                                    >
                                      {layer.type}
                                    </text>
                                  </g>

                                  {/* Neuron count indicator */}
                                  {neuronData.totalNeurons >
                                    neuronData.showing && (
                                    <text
                                      x={neuronData.positions[0].x}
                                      y={
                                        neuronData.positions[
                                          neuronData.positions.length - 1
                                        ].y + 75
                                      }
                                      textAnchor="middle"
                                      className="neuron-count"
                                    >
                                      {neuronData.totalNeurons}n
                                    </text>
                                  )}
                                </g>
                              );
                            })}
                          </svg>
                        );
                      })()}
                    </div>
                  </div>
                )}

              {!analysis && !loading && (
                <div className="empty-state">
                  <Activity className="w-16 h-16 mb-4 opacity-20" />
                  <h3 className="text-lg font-semibold mb-2">
                    Ready to Visualize
                  </h3>
                  <p className="text-sm opacity-60 max-w-md text-center">
                    Paste your TensorFlow, PyTorch, or other ML framework code
                    in the editor. Our AI will analyze and visualize your neural
                    network architecture.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Node Details Modal */}
      <Dialog open={!!selectedNode} onOpenChange={closeModal}>
        <DialogContent className="node-modal" data-testid="node-details-modal">
          <DialogHeader>
            <DialogTitle className="modal-title">
              <div className="modal-header-content">
                <div className="modal-icon">
                  {selectedNode?.type === "Input" && "⭐"}
                  {selectedNode?.type === "Dense" && "🔷"}
                  {selectedNode?.type === "Conv2d" && "🔲"}
                  {selectedNode?.type === "LSTM" && "🔁"}
                  {selectedNode?.type === "Dropout" && "⚡"}
                  {selectedNode?.type === "Output" && "🎯"}
                  {![
                    "Input",
                    "Dense",
                    "Conv2d",
                    "LSTM",
                    "Dropout",
                    "Output",
                  ].includes(selectedNode?.type) && "🔹"}
                </div>
                <div>
                  <h3 className="modal-node-name">{selectedNode?.name}</h3>
                  <Badge className="modal-type-badge">
                    {selectedNode?.type}
                  </Badge>
                </div>
              </div>
            </DialogTitle>
            <button onClick={closeModal} className="modal-close">
              <X className="w-4 h-4" />
            </button>
          </DialogHeader>

          <DialogDescription className="modal-description">
            {selectedNode?.description && (
              <div className="description-section">
                <p>{selectedNode.description}</p>
              </div>
            )}

            <div className="details-grid">
              {selectedNode?.neurons && (
                <div className="detail-item">
                  <span className="detail-label">Neurons/Units:</span>
                  <span className="detail-value">{selectedNode.neurons}</span>
                </div>
              )}

              <div className="detail-item">
                <span className="detail-label">Layer Index:</span>
                <span className="detail-value">
                  {selectedNode?.layer_index + 1}
                </span>
              </div>

              {selectedNode?.params &&
                Object.keys(selectedNode.params).length > 0 && (
                  <div className="params-section">
                    <h4 className="params-title">Parameters</h4>
                    <div className="params-grid">
                      {Object.entries(selectedNode.params).map(
                        ([key, value]) => (
                          <div key={key} className="param-item">
                            <span className="param-key">{key}:</span>
                            <span className="param-value">{String(value)}</span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}
            </div>
          </DialogDescription>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default App;
