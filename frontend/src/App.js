import { useState, useEffect, useCallback } from 'react';
import '@/App.css';
import axios from 'axios';
import Editor from '@monaco-editor/react';
import { Loader2, Play, Download, Share2, Code2, Sparkles, AlertCircle, Info, X, Zap, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable';
import { toast } from 'sonner';

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

  const analyzeCode = useCallback(async (codeToAnalyze) => {
    if (!codeToAnalyze || codeToAnalyze.trim().length === 0) {
      setAnalysis(null);
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/analyze-code`, {
        code: codeToAnalyze,
      });
      setAnalysis(response.data);
      
      if (response.data.success) {
        toast.success('Architecture analyzed with AI!');
      } else if (response.data.error) {
        toast.error(response.data.error);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      toast.error('Failed to analyze code');
      setAnalysis({
        success: false,
        error: error.response?.data?.detail || 'Failed to analyze code',
        layers: [],
        connections: [],
        warnings: []
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
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    const exportFileDefaultName = 'neural-architecture.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    toast.success('Architecture downloaded!');
  };

  const handleShare = async () => {
    if (!analysis || !analysis.success) return;
    
    try {
      await navigator.clipboard.writeText(JSON.stringify(analysis, null, 2));
      toast.success('Copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy');
    }
  };

  const handleNodeClick = (layer) => {
    setSelectedNode(layer);
  };

  const closeModal = () => {
    setSelectedNode(null);
  };

  // Calculate neuron positions for a layer
  const calculateNeuronPositions = (layer, layerIndex, totalLayers) => {
    const neurons = layer.neurons || 4;
    const maxNeuronsToShow = Math.min(neurons, 8);
    const positions = [];
    
    const canvasWidth = 1000;
    const canvasHeight = 500;
    const xPos = (layerIndex / (totalLayers - 1)) * (canvasWidth - 200) + 100;
    const spacing = Math.min(60, canvasHeight / (maxNeuronsToShow + 1));
    const startY = (canvasHeight - (maxNeuronsToShow - 1) * spacing) / 2;
    
    for (let i = 0; i < maxNeuronsToShow; i++) {
      positions.push({
        x: xPos,
        y: startY + i * spacing,
        neuronIndex: i
      });
    }
    
    return { positions, totalNeurons: neurons, showing: maxNeuronsToShow };
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
      <main className="main-content">
        <ResizablePanelGroup direction="horizontal" className="resizable-container">
          {/* Code Editor Panel */}
          <ResizablePanel defaultSize={40} minSize={30}>
            <div className="panel-container editor-panel">
              <div className="panel-header">
                <Code2 className="w-5 h-5" />
                <h2 className="panel-title">Code Editor</h2>
                {analysis?.framework && (
                  <Badge variant="outline" className="ml-auto framework-badge">
                    {analysis.framework}
                  </Badge>
                )}
              </div>
              <div className="editor-container" data-testid="code-editor">
                <Editor
                  height="100%"
                  defaultLanguage="python"
                  value={code}
                  onChange={(value) => setCode(value || '')}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    lineNumbers: 'on',
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    tabSize: 4,
                    padding: { top: 16 },
                  }}
                />
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle className="resize-handle" />

          {/* Visualization Panel */}
          <ResizablePanel defaultSize={60} minSize={40}>
            <div className="panel-container visualization-panel">
              <div className="panel-header">
                <Sparkles className="w-5 h-5" />
                <h2 className="panel-title">Neural Network</h2>
                {loading && (
                  <div className="ml-auto flex items-center text-sm analyzing-badge">
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing with AI...
                  </div>
                )}
              </div>

              <div className="visualization-container" data-testid="visualization-canvas">
                {analysis?.error && (
                  <div className="error-state">
                    <AlertCircle className="w-12 h-12 mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Analysis Error</h3>
                    <p className="text-sm opacity-80">{analysis.error}</p>
                  </div>
                )}

                {analysis?.success && analysis?.layers && analysis.layers.length > 0 && (
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
                      <svg width="100%" height="500" className="neural-svg">
                        <defs>
                          <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style={{ stopColor: '#6366f1', stopOpacity: 0.2 }} />
                            <stop offset="100%" style={{ stopColor: '#8b5cf6', stopOpacity: 0.2 }} />
                          </linearGradient>
                          <filter id="glow">
                            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                            <feMerge>
                              <feMergeNode in="coloredBlur"/>
                              <feMergeNode in="SourceGraphic"/>
                            </feMerge>
                          </filter>
                        </defs>

                        {/* Draw connections between neurons */}
                        {analysis.layers.map((layer, layerIdx) => {
                          if (layerIdx === analysis.layers.length - 1) return null;
                          
                          const currentLayer = calculateNeuronPositions(layer, layerIdx, analysis.layers.length);
                          const nextLayer = calculateNeuronPositions(analysis.layers[layerIdx + 1], layerIdx + 1, analysis.layers.length);
                          
                          return currentLayer.positions.map((sourcePos, sourceIdx) => (
                            nextLayer.positions.map((targetPos, targetIdx) => (
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
                            ))
                          ));
                        })}

                        {/* Draw neurons */}
                        {analysis.layers.map((layer, layerIdx) => {
                          const neuronData = calculateNeuronPositions(layer, layerIdx, analysis.layers.length);
                          
                          return (
                            <g key={`layer-${layerIdx}`}>
                              {neuronData.positions.map((pos, neuronIdx) => (
                                <circle
                                  key={`neuron-${layerIdx}-${neuronIdx}`}
                                  cx={pos.x}
                                  cy={pos.y}
                                  r="20"
                                  className={`neuron ${
                                    hoveredNode?.id === layer.id ? 'hovered' : ''
                                  } ${
                                    selectedNode?.id === layer.id ? 'selected' : ''
                                  }`}
                                  onClick={() => handleNodeClick(layer)}
                                  onMouseEnter={() => setHoveredNode(layer)}
                                  onMouseLeave={() => setHoveredNode(null)}
                                  data-testid={`neuron-${layerIdx}-${neuronIdx}`}
                                  filter={selectedNode?.id === layer.id || hoveredNode?.id === layer.id ? 'url(#glow)' : ''}
                                />
                              ))}
                              
                              {/* Layer label */}
                              <text
                                x={neuronData.positions[0].x}
                                y={neuronData.positions[neuronData.positions.length - 1].y + 50}
                                textAnchor="middle"
                                className="layer-label"
                                onClick={() => handleNodeClick(layer)}
                              >
                                {layer.name}
                              </text>
                              
                              {/* Neuron count indicator */}
                              {neuronData.totalNeurons > neuronData.showing && (
                                <text
                                  x={neuronData.positions[0].x}
                                  y={neuronData.positions[neuronData.positions.length - 1].y + 70}
                                  textAnchor="middle"
                                  className="neuron-count"
                                >
                                  ({neuronData.totalNeurons} neurons)
                                </text>
                              )}
                            </g>
                          );
                        })}
                      </svg>
                    </div>
                  </div>
                )}

                {!analysis && !loading && (
                  <div className="empty-state">
                    <Activity className="w-16 h-16 mb-4 opacity-20" />
                    <h3 className="text-lg font-semibold mb-2">Ready to Visualize</h3>
                    <p className="text-sm opacity-60 max-w-md text-center">
                      Paste your TensorFlow, PyTorch, or Scikit-learn code in the editor.
                      Our AI will analyze and visualize your neural network architecture.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </main>

      {/* Node Details Modal */}
      <Dialog open={!!selectedNode} onOpenChange={closeModal}>
        <DialogContent className="node-modal" data-testid="node-details-modal">
          <DialogHeader>
            <DialogTitle className="modal-title">
              <div className="modal-header-content">
                <div className="modal-icon">
                  {selectedNode?.type === 'Input' && '⭐'}
                  {selectedNode?.type === 'Dense' && '🔷'}
                  {selectedNode?.type === 'Conv2d' && '🔲'}
                  {selectedNode?.type === 'LSTM' && '🔁'}
                  {selectedNode?.type === 'Dropout' && '⚡'}
                  {selectedNode?.type === 'Output' && '🎯'}
                  {!['Input', 'Dense', 'Conv2d', 'LSTM', 'Dropout', 'Output'].includes(selectedNode?.type) && '🔹'}
                </div>
                <div>
                  <h3 className="modal-node-name">{selectedNode?.name}</h3>
                  <Badge className="modal-type-badge">{selectedNode?.type}</Badge>
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
                <span className="detail-value">{selectedNode?.layer_index + 1}</span>
              </div>
              
              {selectedNode?.params && Object.keys(selectedNode.params).length > 0 && (
                <div className="params-section">
                  <h4 className="params-title">Parameters</h4>
                  <div className="params-grid">
                    {Object.entries(selectedNode.params).map(([key, value]) => (
                      <div key={key} className="param-item">
                        <span className="param-key">{key}:</span>
                        <span className="param-value">{String(value)}</span>
                      </div>
                    ))}
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