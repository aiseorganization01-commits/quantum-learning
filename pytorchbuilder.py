import React, { useState } from 'react';
import { Trash2, Play, Download, Database, RotateCcw } from 'lucide-react';

const PyTorchBlockBuilder = () => {
  const [blocks, setBlocks] = useState([]);
  const [draggedBlock, setDraggedBlock] = useState(null);
  const [generatedCode, setGeneratedCode] = useState('');
  const [showCode, setShowCode] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('mnist');

  const datasets = {
    mnist: {
      name: 'MNIST Digits',
      description: 'Handwritten digits (28x28 grayscale)',
      inputShape: '1x28x28',
      outputClasses: 10,
      color: 'bg-cyan-600'
    },
    cifar10: {
      name: 'CIFAR-10',
      description: 'Color images (32x32 RGB)',
      inputShape: '3x32x32',
      outputClasses: 10,
      color: 'bg-teal-600'
    },
    fashion: {
      name: 'Fashion-MNIST',
      description: 'Fashion items (28x28 grayscale)',
      inputShape: '1x28x28',
      outputClasses: 10,
      color: 'bg-indigo-600'
    },
    iris: {
      name: 'Iris Dataset',
      description: 'Flower measurements (4 features)',
      inputShape: '4',
      outputClasses: 3,
      color: 'bg-pink-600'
    }
  };

  const blockTypes = {
    input: [
      { id: 'input', name: 'Input', icon: '📥', color: 'bg-gradient-to-br from-cyan-500 to-blue-500', params: ['size'], shape: 'hexagon' }
    ],
    conv: [
      { id: 'conv2d', name: 'Conv2D', icon: '🔲', color: 'bg-gradient-to-br from-blue-500 to-blue-600', params: ['out_channels', 'kernel_size', 'stride', 'padding'], shape: 'puzzle-top' },
      { id: 'conv2d_3x3', name: 'Conv2D 3×3', icon: '⊞', color: 'bg-gradient-to-br from-blue-400 to-blue-500', params: ['out_channels'], shape: 'puzzle-top' },
      { id: 'conv2d_5x5', name: 'Conv2D 5×5', icon: '⊡', color: 'bg-gradient-to-br from-blue-600 to-blue-700', params: ['out_channels'], shape: 'puzzle-top' },
      { id: 'conv1d', name: 'Conv1D', icon: '▬', color: 'bg-gradient-to-br from-blue-300 to-blue-400', params: ['out_channels', 'kernel_size'], shape: 'puzzle-top' },
    ],
    pooling: [
      { id: 'maxpool2d', name: 'MaxPool2D', icon: '⬇️', color: 'bg-gradient-to-br from-purple-500 to-purple-600', params: ['kernel_size', 'stride'], shape: 'puzzle-mid' },
      { id: 'avgpool2d', name: 'AvgPool2D', icon: '📊', color: 'bg-gradient-to-br from-purple-400 to-purple-500', params: ['kernel_size'], shape: 'puzzle-mid' },
      { id: 'adaptiveavg', name: 'AdaptiveAvg', icon: '🎯', color: 'bg-gradient-to-br from-purple-600 to-purple-700', params: ['output_size'], shape: 'puzzle-mid' },
      { id: 'maxpool1d', name: 'MaxPool1D', icon: '⤓', color: 'bg-gradient-to-br from-purple-300 to-purple-400', params: ['kernel_size'], shape: 'puzzle-mid' },
    ],
    linear: [
      { id: 'flatten', name: 'Flatten', icon: '▭', color: 'bg-gradient-to-br from-orange-500 to-orange-600', params: [], shape: 'puzzle-mid' },
      { id: 'linear', name: 'Linear', icon: '➡️', color: 'bg-gradient-to-br from-green-500 to-green-600', params: ['out_features'], shape: 'puzzle-mid' },
      { id: 'linear_small', name: 'Linear 128', icon: '→', color: 'bg-gradient-to-br from-green-400 to-green-500', params: [], shape: 'puzzle-mid' },
      { id: 'linear_large', name: 'Linear 512', icon: '⇒', color: 'bg-gradient-to-br from-green-600 to-green-700', params: [], shape: 'puzzle-mid' },
    ],
    activation: [
      { id: 'relu', name: 'ReLU', icon: '📈', color: 'bg-gradient-to-br from-lime-500 to-lime-600', params: [], shape: 'puzzle-mid' },
      { id: 'leakyrelu', name: 'LeakyReLU', icon: '📉', color: 'bg-gradient-to-br from-lime-400 to-lime-500', params: ['negative_slope'], shape: 'puzzle-mid' },
      { id: 'sigmoid', name: 'Sigmoid', icon: '〰️', color: 'bg-gradient-to-br from-lime-600 to-lime-700', params: [], shape: 'puzzle-mid' },
      { id: 'tanh', name: 'Tanh', icon: '∿', color: 'bg-gradient-to-br from-lime-500 to-green-600', params: [], shape: 'puzzle-mid' },
      { id: 'softmax', name: 'Softmax', icon: '🎲', color: 'bg-gradient-to-br from-lime-700 to-green-700', params: ['dim'], shape: 'puzzle-mid' },
      { id: 'elu', name: 'ELU', icon: '📐', color: 'bg-gradient-to-br from-lime-300 to-lime-400', params: ['alpha'], shape: 'puzzle-mid' },
      { id: 'gelu', name: 'GELU', icon: '🌊', color: 'bg-gradient-to-br from-lime-600 to-teal-600', params: [], shape: 'puzzle-mid' },
      { id: 'selu', name: 'SELU', icon: '⚡', color: 'bg-gradient-to-br from-lime-400 to-green-500', params: [], shape: 'puzzle-mid' },
    ],
    regularization: [
      { id: 'dropout', name: 'Dropout', icon: '💧', color: 'bg-gradient-to-br from-red-500 to-red-600', params: ['p'], shape: 'puzzle-mid' },
      { id: 'dropout2d', name: 'Dropout2D', icon: '💦', color: 'bg-gradient-to-br from-red-400 to-red-500', params: ['p'], shape: 'puzzle-mid' },
      { id: 'batchnorm1d', name: 'BatchNorm1D', icon: '📏', color: 'bg-gradient-to-br from-yellow-500 to-yellow-600', params: ['num_features'], shape: 'puzzle-mid' },
      { id: 'batchnorm2d', name: 'BatchNorm2D', icon: '📐', color: 'bg-gradient-to-br from-yellow-600 to-yellow-700', params: ['num_features'], shape: 'puzzle-mid' },
      { id: 'layernorm', name: 'LayerNorm', icon: '📊', color: 'bg-gradient-to-br from-yellow-400 to-yellow-500', params: ['normalized_shape'], shape: 'puzzle-mid' },
    ],
    tensor_ops: [
      { id: 'reshape', name: 'Reshape', icon: '🔄', color: 'bg-gradient-to-br from-cyan-500 to-cyan-600', params: ['target_shape'], shape: 'puzzle-mid' },
      { id: 'transpose', name: 'Transpose', icon: '↔️', color: 'bg-gradient-to-br from-cyan-400 to-cyan-500', params: ['dim0', 'dim1'], shape: 'puzzle-mid' },
      { id: 'squeeze', name: 'Squeeze', icon: '⊡', color: 'bg-gradient-to-br from-cyan-600 to-cyan-700', params: [], shape: 'puzzle-mid' },
      { id: 'unsqueeze', name: 'Unsqueeze', icon: '⊞', color: 'bg-gradient-to-br from-cyan-300 to-cyan-400', params: ['dim'], shape: 'puzzle-mid' },
      { id: 'cat', name: 'Concatenate', icon: '🔗', color: 'bg-gradient-to-br from-cyan-700 to-blue-600', params: ['dim'], shape: 'puzzle-mid' },
    ],
    math_ops: [
      { id: 'add', name: 'Add', icon: '➕', color: 'bg-gradient-to-br from-pink-500 to-pink-600', params: ['value'], shape: 'puzzle-mid' },
      { id: 'multiply', name: 'Multiply', icon: '✖️', color: 'bg-gradient-to-br from-pink-400 to-pink-500', params: ['value'], shape: 'puzzle-mid' },
      { id: 'clamp', name: 'Clamp', icon: '📌', color: 'bg-gradient-to-br from-pink-600 to-pink-700', params: ['min_val', 'max_val'], shape: 'puzzle-mid' },
      { id: 'normalize', name: 'Normalize', icon: '⚖️', color: 'bg-gradient-to-br from-pink-300 to-pink-400', params: [], shape: 'puzzle-mid' },
    ],
    numpy_ops: [
      { id: 'np_mean', name: 'Mean', icon: '📊', color: 'bg-gradient-to-br from-teal-500 to-teal-600', params: ['axis'], shape: 'puzzle-mid' },
      { id: 'np_std', name: 'Std Dev', icon: '📈', color: 'bg-gradient-to-br from-teal-400 to-teal-500', params: ['axis'], shape: 'puzzle-mid' },
      { id: 'np_sum', name: 'Sum', icon: '∑', color: 'bg-gradient-to-br from-teal-600 to-teal-700', params: ['axis'], shape: 'puzzle-mid' },
      { id: 'np_max', name: 'Max', icon: '⬆️', color: 'bg-gradient-to-br from-teal-300 to-teal-400', params: ['axis'], shape: 'puzzle-mid' },
      { id: 'np_min', name: 'Min', icon: '⬇️', color: 'bg-gradient-to-br from-teal-700 to-cyan-700', params: ['axis'], shape: 'puzzle-mid' },
    ],
    output: [
      { id: 'output', name: 'Output', icon: '📤', color: 'bg-gradient-to-br from-pink-500 to-red-500', params: ['classes'], shape: 'puzzle-bottom' }
    ],
  };

  const handleDragStart = (e, blockType, category) => {
    setDraggedBlock({ ...blockType, category });
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (draggedBlock) {
      const newBlock = {
        ...draggedBlock,
        id: `${draggedBlock.id}_${Date.now()}`,
        values: {}
      };
      draggedBlock.params.forEach(param => {
        newBlock.values[param] = getDefaultValue(param, draggedBlock.id);
      });
      setBlocks([...blocks, newBlock]);
      setDraggedBlock(null);
    }
  };

  const getDefaultValue = (param, blockId) => {
    const defaults = {
      'size': datasets[selectedDataset].inputShape,
      'out_channels': 32,
      'kernel_size': 3,
      'stride': 1,
      'padding': 1,
      'out_features': 128,
      'num_features': 64,
      'p': 0.5,
      'negative_slope': 0.01,
      'dim': 1,
      'output_size': 1,
      'classes': datasets[selectedDataset].outputClasses,
      'alpha': 1.0,
      'target_shape': '-1',
      'dim0': 0,
      'dim1': 1,
      'value': 1.0,
      'min_val': -1.0,
      'max_val': 1.0,
      'axis': -1,
      'normalized_shape': 128
    };
    if (blockId === 'linear_small') return 128;
    if (blockId === 'linear_large') return 512;
    if (blockId === 'conv2d_3x3') return 32;
    if (blockId === 'conv2d_5x5') return 64;
    return defaults[param] || 1;
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const removeBlock = (blockId) => {
    setBlocks(blocks.filter(b => b.id !== blockId));
  };

  const updateBlockValue = (blockId, param, value) => {
    setBlocks(blocks.map(block =>
      block.id === blockId
        ? { ...block, values: { ...block.values, [param]: value } }
        : block
    ));
  };

  const generateCode = () => {
    let code = `import torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torchvision\nimport torchvision.transforms as transforms\n\n`;
    
    code += `# Load ${datasets[selectedDataset].name}\n`;
    if (selectedDataset === 'mnist') {
      code += `transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])\n`;
      code += `trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n`;
      code += `testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)\n`;
    } else if (selectedDataset === 'cifar10') {
      code += `transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n`;
      code += `trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)\n`;
      code += `testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)\n`;
    } else if (selectedDataset === 'fashion') {
      code += `transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])\n`;
      code += `trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)\n`;
      code += `testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)\n`;
    }
    code += `trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)\n`;
    code += `testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)\n\n`;
    
    code += `class MyModel(nn.Module):\n`;
    code += `    def __init__(self):\n`;
    code += `        super(MyModel, self).__init__()\n`;
    
    const layerBlocks = blocks.filter(b => !['input', 'output'].includes(b.id.split('_')[0]));
    layerBlocks.forEach((block, idx) => {
      const blockType = block.id.split('_')[0];
      if (blockType === 'conv2d') {
        const inChannels = idx === 0 ? (selectedDataset === 'cifar10' ? 3 : 1) : 32;
        if (block.params.length > 1) {
          code += `        self.layer${idx + 1} = nn.Conv2d(${inChannels}, ${block.values.out_channels}, kernel_size=${block.values.kernel_size}, stride=${block.values.stride}, padding=${block.values.padding})\n`;
        } else {
          const kernel = block.id.includes('3x3') ? 3 : 5;
          code += `        self.layer${idx + 1} = nn.Conv2d(${inChannels}, ${block.values.out_channels}, kernel_size=${kernel}, padding=${Math.floor(kernel/2)})\n`;
        }
      } else if (blockType === 'conv1d') {
        code += `        self.layer${idx + 1} = nn.Conv1d(1, ${block.values.out_channels}, kernel_size=${block.values.kernel_size})\n`;
      } else if (blockType === 'maxpool2d') {
        code += `        self.layer${idx + 1} = nn.MaxPool2d(kernel_size=${block.values.kernel_size}, stride=${block.values.stride})\n`;
      } else if (blockType === 'maxpool1d') {
        code += `        self.layer${idx + 1} = nn.MaxPool1d(kernel_size=${block.values.kernel_size})\n`;
      } else if (blockType === 'avgpool2d') {
        code += `        self.layer${idx + 1} = nn.AvgPool2d(kernel_size=${block.values.kernel_size})\n`;
      } else if (blockType === 'adaptiveavg') {
        code += `        self.layer${idx + 1} = nn.AdaptiveAvgPool2d(${block.values.output_size})\n`;
      } else if (blockType === 'linear') {
        const outFeatures = block.params.length > 0 ? block.values.out_features : (block.id.includes('small') ? 128 : 512);
        code += `        self.layer${idx + 1} = nn.Linear(512, ${outFeatures})  # Adjust input size based on previous layers\n`;
      } else if (blockType === 'dropout') {
        code += `        self.layer${idx + 1} = nn.Dropout(p=${block.values.p})\n`;
      } else if (blockType === 'dropout2d') {
        code += `        self.layer${idx + 1} = nn.Dropout2d(p=${block.values.p})\n`;
      } else if (blockType === 'batchnorm1d') {
        code += `        self.layer${idx + 1} = nn.BatchNorm1d(${block.values.num_features})\n`;
      } else if (blockType === 'batchnorm2d') {
        code += `        self.layer${idx + 1} = nn.BatchNorm2d(${block.values.num_features})\n`;
      } else if (blockType === 'layernorm') {
        code += `        self.layer${idx + 1} = nn.LayerNorm(${block.values.normalized_shape})\n`;
      } else if (blockType === 'flatten') {
        code += `        self.layer${idx + 1} = nn.Flatten()\n`;
      }
    });
    
    const outputBlock = blocks.find(b => b.id.startsWith('output'));
    if (outputBlock) {
      code += `        self.output = nn.Linear(128, ${outputBlock.values.classes})\n`;
    }
    
    code += `\n    def forward(self, x):\n`;
    blocks.filter(b => !['input', 'output'].includes(b.id.split('_')[0])).forEach((block, idx) => {
      const blockType = block.id.split('_')[0];
      if (['conv2d', 'conv1d', 'maxpool2d', 'maxpool1d', 'avgpool2d', 'adaptiveavg', 'linear', 'dropout', 'dropout2d', 'batchnorm1d', 'batchnorm2d', 'layernorm', 'flatten'].includes(blockType)) {
        code += `        x = self.layer${idx + 1}(x)\n`;
      } else if (blockType === 'relu') {
        code += `        x = torch.relu(x)\n`;
      } else if (blockType === 'leakyrelu') {
        code += `        x = torch.nn.functional.leaky_relu(x, negative_slope=${block.values.negative_slope})\n`;
      } else if (blockType === 'elu') {
        code += `        x = torch.nn.functional.elu(x, alpha=${block.values.alpha})\n`;
      } else if (blockType === 'gelu') {
        code += `        x = torch.nn.functional.gelu(x)\n`;
      } else if (blockType === 'selu') {
        code += `        x = torch.nn.functional.selu(x)\n`;
      } else if (blockType === 'sigmoid') {
        code += `        x = torch.sigmoid(x)\n`;
      } else if (blockType === 'tanh') {
        code += `        x = torch.tanh(x)\n`;
      } else if (blockType === 'softmax') {
        code += `        x = torch.softmax(x, dim=${block.values.dim})\n`;
      } else if (blockType === 'reshape') {
        code += `        x = x.reshape(${block.values.target_shape})\n`;
      } else if (blockType === 'transpose') {
        code += `        x = x.transpose(${block.values.dim0}, ${block.values.dim1})\n`;
      } else if (blockType === 'squeeze') {
        code += `        x = x.squeeze()\n`;
      } else if (blockType === 'unsqueeze') {
        code += `        x = x.unsqueeze(${block.values.dim})\n`;
      } else if (blockType === 'cat') {
        code += `        # x = torch.cat([x, other_tensor], dim=${block.values.dim})\n`;
      } else if (blockType === 'add') {
        code += `        x = x + ${block.values.value}\n`;
      } else if (blockType === 'multiply') {
        code += `        x = x * ${block.values.value}\n`;
      } else if (blockType === 'clamp') {
        code += `        x = torch.clamp(x, min=${block.values.min_val}, max=${block.values.max_val})\n`;
      } else if (blockType === 'normalize') {
        code += `        x = torch.nn.functional.normalize(x, p=2, dim=1)\n`;
      } else if (blockType === 'np') {
        const npOp = block.id.split('_')[1];
        if (npOp === 'mean') {
          code += `        x = torch.mean(x, dim=${block.values.axis})\n`;
        } else if (npOp === 'std') {
          code += `        x = torch.std(x, dim=${block.values.axis})\n`;
        } else if (npOp === 'sum') {
          code += `        x = torch.sum(x, dim=${block.values.axis})\n`;
        } else if (npOp === 'max') {
          code += `        x = torch.max(x, dim=${block.values.axis})[0]\n`;
        } else if (npOp === 'min') {
          code += `        x = torch.min(x, dim=${block.values.axis})[0]\n`;
        }
      }
    });
    
    if (outputBlock) {
      code += `        x = self.output(x)\n`;
    }
    code += `        return x\n\n`;
    
    code += `# Initialize model\n`;
    code += `model = MyModel()\n`;
    code += `criterion = nn.CrossEntropyLoss()\n`;
    code += `optimizer = optim.Adam(model.parameters(), lr=0.001)\n\n`;
    
    code += `# Training loop\n`;
    code += `for epoch in range(10):\n`;
    code += `    running_loss = 0.0\n`;
    code += `    for i, data in enumerate(trainloader, 0):\n`;
    code += `        inputs, labels = data\n`;
    code += `        optimizer.zero_grad()\n`;
    code += `        outputs = model(inputs)\n`;
    code += `        loss = criterion(outputs, labels)\n`;
    code += `        loss.backward()\n`;
    code += `        optimizer.step()\n`;
    code += `        running_loss += loss.item()\n`;
    code += `    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')\n`;
    
    setGeneratedCode(code);
    setShowCode(true);
  };

  const downloadCode = () => {
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pytorch_model.py';
    a.click();
  };

  const getPuzzleStyle = (shape) => {
    return "relative";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <span>🧩</span> PyTorch Puzzle Builder
          </h1>
          <p className="text-purple-200 text-lg">Snap blocks together like puzzle pieces!</p>
        </div>

        {/* Dataset Selector */}
        <div className="mb-6 bg-slate-800 bg-opacity-50 backdrop-blur-sm rounded-xl p-6 border border-purple-500 border-opacity-30">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <Database className="w-6 h-6" /> Choose Your Dataset
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {Object.entries(datasets).map(([key, dataset]) => (
              <button
                key={key}
                onClick={() => setSelectedDataset(key)}
                className={`${dataset.color} ${
                  selectedDataset === key ? 'ring-4 ring-white scale-105' : 'opacity-60 hover:opacity-80'
                } text-white p-4 rounded-lg transition-all transform`}
              >
                <div className="font-bold text-lg mb-1">{dataset.name}</div>
                <div className="text-sm opacity-90 mb-2">{dataset.description}</div>
                <div className="text-xs font-mono bg-black bg-opacity-30 rounded px-2 py-1">
                  Input: {dataset.inputShape} | Classes: {dataset.outputClasses}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Block Palette */}
          <div className="lg:col-span-1 space-y-4">
            {Object.entries(blockTypes).map(([category, items]) => (
              <div key={category} className="bg-slate-800 bg-opacity-50 backdrop-blur-sm rounded-xl p-4 border border-purple-500 border-opacity-30">
                <h3 className="text-white font-bold mb-3 uppercase text-sm tracking-wider">{category}</h3>
                <div className="space-y-2">
                  {items.map((block) => (
                    <div
                      key={block.id}
                      draggable
                      onDragStart={(e) => handleDragStart(e, block, category)}
                      className={`${block.color} ${getPuzzleStyle(block.shape)} text-white p-3 rounded-lg cursor-move hover:scale-105 transition-transform shadow-lg border-2 border-white border-opacity-30`}
                    >
                      <div className="flex items-center gap-2 font-bold">
                        <span className="text-xl">{block.icon}</span>
                        <span className="text-sm">{block.name}</span>
                      </div>
                      {block.params.length > 0 && (
                        <div className="text-xs mt-1 opacity-75">
                          {block.params.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Build Area */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800 bg-opacity-50 backdrop-blur-sm rounded-xl p-6 border border-purple-500 border-opacity-30 min-h-96">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                  🏗️ Build Your Network
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={generateCode}
                    disabled={blocks.length === 0}
                    className="flex items-center gap-2 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:from-slate-600 disabled:to-slate-700 text-white px-4 py-2 rounded-lg transition-all shadow-lg font-semibold"
                  >
                    <Play className="w-4 h-4" /> Generate Code
                  </button>
                  <button
                    onClick={() => setBlocks([])}
                    disabled={blocks.length === 0}
                    className="flex items-center gap-2 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 disabled:from-slate-600 disabled:to-slate-700 text-white px-4 py-2 rounded-lg transition-all shadow-lg font-semibold"
                  >
                    <RotateCcw className="w-4 h-4" /> Reset
                  </button>
                </div>
              </div>

              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="border-2 border-dashed border-purple-400 rounded-lg p-6 min-h-80 bg-slate-900 bg-opacity-30"
              >
                {blocks.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-purple-300 text-center py-20">
                    <div className="text-6xl mb-4">🧩</div>
                    <div className="text-xl font-semibold mb-2">Drag puzzle pieces here!</div>
                    <div className="text-sm opacity-75">Start with an Input block</div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {blocks.map((block, idx) => (
                      <div
                        key={block.id}
                        className={`${block.color} text-white p-4 rounded-lg shadow-xl border-2 border-white border-opacity-40 group relative`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="bg-white bg-opacity-20 rounded-full w-8 h-8 flex items-center justify-center font-bold text-sm">
                              {idx + 1}
                            </div>
                            <span className="text-2xl">{block.icon}</span>
                            <span className="font-bold text-lg">{block.name}</span>
                          </div>
                          <button
                            onClick={() => removeBlock(block.id)}
                            className="bg-red-500 hover:bg-red-600 p-2 rounded-lg transition-colors opacity-0 group-hover:opacity-100 shadow-lg"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                        {block.params.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-3 ml-11">
                            {block.params.map(param => (
                              <div key={param} className="flex items-center gap-2">
                                <label className="text-sm font-semibold opacity-90">{param}:</label>
                                <input
                                  type="number"
                                  value={block.values[param]}
                                  onChange={(e) => updateBlockValue(block.id, param, parseFloat(e.target.value) || 0)}
                                  className="bg-white text-slate-900 px-2 py-1 rounded w-16 text-sm font-bold"
                                />
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Generated Code */}
            {showCode && (
              <div className="mt-6 bg-slate-800 bg-opacity-50 backdrop-blur-sm rounded-xl p-6 border border-purple-500 border-opacity-30">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                    💻 Your PyTorch Code
                  </h2>
                  <button
                    onClick={downloadCode}
                    className="flex items-center gap-2 bg-gradient-to-r from-blue-500 to-cyan-600 hover:from-blue-600 hover:to-cyan-700 text-white px-4 py-2 rounded-lg transition-all shadow-lg font-semibold"
                  >
                    <Download className="w-4 h-4" /> Download .py
                  </button>
                </div>
                <pre className="bg-slate-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm font-mono border border-slate-700">
                  {generatedCode}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PyTorchBlockBuilder;
