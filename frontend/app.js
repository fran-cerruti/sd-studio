// ============================================================================
// CONFIGURACIÓN Y UTILIDADES
// ============================================================================

// Detectar automáticamente el host (funciona en localhost y red local)
const API_BASE = `http://${window.location.hostname}:8080`;
const { createApp } = Vue;

const utils = {
    countTokens(text) {
        if (!text) return 0;
        
        // Aproximación mejorada del tokenizador CLIP
        // Basado en análisis de comportamiento real del tokenizador
        
        // 1. Contar palabras base (split por espacios y comas)
        let words = text.split(/[\s,]+/).filter(t => t.length > 0);
        let tokenCount = words.length;
        
        // 2. Ajustar por palabras compuestas largas
        // Palabras >10 caracteres suelen dividirse en múltiples tokens
        words.forEach(word => {
            if (word.length > 10) {
                // Aproximación: +0.5 tokens por cada 10 caracteres adicionales
                tokenCount += Math.floor((word.length - 10) / 10) * 0.5;
            }
        });
        
        // 3. Contar números y caracteres especiales
        const numbers = text.match(/\d+/g) || [];
        tokenCount += numbers.length * 0.3; // Números suelen ser tokens separados
        
        // 4. Agregar tokens especiales [BOS] y [EOS]
        tokenCount += 2;
        
        // 5. Redondear hacia arriba (ser conservador)
        return Math.ceil(tokenCount);
    },
    
    getTokenClass(count) {
        if (count > 77) return 'text-accent-danger';
        if (count >= 75) return 'text-accent-warning';
        return 'text-accent-success';
    },
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    },
    
    async logGeneration(data) {
        try {
            await axios.post(`${API_BASE}/log`, {
                timestamp: new Date().toISOString(),
                ...data
            });
        } catch (error) {
            console.error('Error logging generation:', error);
        }
    }
};

// ============================================================================
// COMPONENTE REUTILIZABLE: PROMPT EDITOR
// ============================================================================

const PromptEditor = {
    template: `
        <div class="space-y-4">
            <!-- Saved Prompts Selector -->
            <div class="card">
                <div class="flex gap-2">
                    <select 
                        v-model="selectedPromptIndex" 
                        @change="loadSelectedPrompt"
                        class="flex-1 px-3 py-2 bg-dark-card border border-dark-border rounded text-dark-text"
                    >
                        <option :value="null">-- Select saved prompt --</option>
                        <option v-for="(p, i) in savedPrompts" :key="i" :value="i">
                            {{ p.title }}
                        </option>
                    </select>
                    <button @click="showSaveDialog = true" class="btn btn-secondary px-4">
                        Save Current
                    </button>
                </div>
            </div>

            <!-- Prompts -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div class="card">
                    <div class="flex justify-between items-center mb-2">
                        <label class="font-medium">Positive Prompt</label>
                        <span :class="['text-sm font-mono', utils.getTokenClass(positiveTokens)]">
                            {{ positiveTokens }}/77
                        </span>
                    </div>
                    <textarea 
                        v-model="localPrompt" 
                        @input="updateTokens"
                        rows="4" 
                        class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text resize-none focus:outline-none focus:border-accent-primary"
                        placeholder="Describe what you want to generate..."
                    ></textarea>
                </div>
                
                <div class="card">
                    <div class="flex justify-between items-center mb-2">
                        <label class="font-medium">Negative Prompt</label>
                        <span :class="['text-sm font-mono', utils.getTokenClass(negativeTokens)]">
                            {{ negativeTokens }}/77
                        </span>
                    </div>
                    <textarea 
                        v-model="localNegativePrompt" 
                        @input="updateTokens"
                        rows="4" 
                        class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text resize-none focus:outline-none focus:border-accent-primary"
                        placeholder="What to avoid..."
                    ></textarea>
                </div>
            </div>

            <!-- Save Dialog -->
            <div v-if="showSaveDialog" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" @click.self="showSaveDialog = false">
                <div class="bg-dark-card border border-dark-border rounded-lg p-6 max-w-md w-full mx-4">
                    <h3 class="text-xl font-bold mb-4">Save Prompt</h3>
                    <input 
                        v-model="saveTitle" 
                        type="text" 
                        class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text mb-4"
                        placeholder="Enter a title..."
                        @keyup.enter="savePrompt"
                    >
                    <div class="flex gap-2">
                        <button @click="savePrompt" class="btn btn-primary flex-1">Save</button>
                        <button @click="showSaveDialog = false" class="btn btn-secondary flex-1">Cancel</button>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['savedPrompts', 'prompt', 'negativePrompt'],
    emits: ['update:prompt', 'update:negativePrompt', 'save-prompt'],
    data() {
        return {
            utils,
            localPrompt: this.prompt || '',
            localNegativePrompt: this.negativePrompt || '',
            positiveTokens: 0,
            negativeTokens: 0,
            selectedPromptIndex: null,
            showSaveDialog: false,
            saveTitle: ''
        };
    },
    watch: {
        prompt(newVal) {
            this.localPrompt = newVal;
            this.updateTokens();
        },
        negativePrompt(newVal) {
            this.localNegativePrompt = newVal;
            this.updateTokens();
        },
        localPrompt(newVal) {
            this.$emit('update:prompt', newVal);
        },
        localNegativePrompt(newVal) {
            this.$emit('update:negativePrompt', newVal);
        }
    },
    mounted() {
        this.updateTokens();
    },
    methods: {
        updateTokens() {
            this.positiveTokens = utils.countTokens(this.localPrompt);
            this.negativeTokens = utils.countTokens(this.localNegativePrompt);
        },
        loadSelectedPrompt() {
            if (this.selectedPromptIndex !== null) {
                const selected = this.savedPrompts[this.selectedPromptIndex];
                this.localPrompt = selected.positive;
                this.localNegativePrompt = selected.negative;
            }
        },
        savePrompt() {
            if (!this.saveTitle.trim()) {
                alert('Please enter a title');
                return;
            }
            this.$emit('save-prompt', {
                title: this.saveTitle,
                positive: this.localPrompt,
                negative: this.localNegativePrompt
            });
            this.saveTitle = '';
            this.showSaveDialog = false;
        }
    }
};

// ============================================================================
// TAB: GENERATE
// ============================================================================

const GenerateTab = {
    template: `
        <div class="space-y-4">
            <!-- Model Selection -->
            <div class="card">
                <h3 class="font-semibold mb-3">Model</h3>
                <div class="flex gap-2">
                    <select 
                        v-model="selectedModel" 
                        class="flex-1 px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text"
                        :disabled="loading"
                    >
                        <option value="">-- Select model --</option>
                        <optgroup label="Local Models">
                            <option v-for="m in models.local" :key="m.id" :value="m.id">
                                {{ m.name }}
                            </option>
                        </optgroup>
                        <optgroup label="HuggingFace Models">
                            <option v-for="m in models.huggingface" :key="m.id" :value="m.id">
                                {{ m.name }}
                            </option>
                        </optgroup>
                    </select>
                    <button 
                        @click="loadModel" 
                        :disabled="!selectedModel || loading"
                        class="btn btn-primary px-6"
                    >
                        {{ loading ? 'Loading...' : 'Load' }}
                    </button>
                </div>
            </div>

            <!-- Prompt Editor -->
            <prompt-editor
                :saved-prompts="savedPrompts"
                v-model:prompt="prompt"
                v-model:negative-prompt="negativePrompt"
                @save-prompt="$emit('save-prompt', $event)"
            ></prompt-editor>

            <!-- Parameters -->
            <div class="card">
                <h3 class="font-semibold mb-3">Parameters</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                        <label class="block text-sm mb-1">Width</label>
                        <input v-model.number="params.width" type="number" step="64" class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text">
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Height</label>
                        <input v-model.number="params.height" type="number" step="64" class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text">
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Steps</label>
                        <input v-model.number="params.steps" type="number" class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text">
                    </div>
                    <div>
                        <label class="block text-sm mb-1">CFG Scale</label>
                        <input v-model.number="params.cfg_scale" type="number" step="0.5" class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text">
                    </div>
                </div>
            </div>

            <!-- Generate Button -->
            <button 
                @click="generate" 
                :disabled="!canGenerate"
                class="btn btn-primary w-full py-3 text-lg font-semibold"
            >
                {{ generating ? 'Generating...' : 'GENERATE' }}
            </button>

            <!-- Results -->
            <div v-if="results.length > 0" class="card">
                <h3 class="font-semibold mb-3">Results</h3>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div v-for="(img, i) in results" :key="i" class="cursor-pointer hover:opacity-80 transition" @click="showImage(img)">
                        <img :src="getImageUrl(img)" class="w-full h-auto rounded border border-dark-border">
                    </div>
                </div>
            </div>

            <!-- Modal -->
            <div v-if="modalImage" class="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50 p-4" @click="modalImage = null">
                <img :src="getImageUrl(modalImage)" class="max-w-full max-h-full object-contain" @click.stop>
                <button @click="modalImage = null" class="absolute top-4 right-4 text-white text-4xl hover:text-accent-primary">&times;</button>
            </div>
        </div>
    `,
    props: ['savedPrompts'],
    emits: ['save-prompt', 'update-gpu'],
    components: { 'prompt-editor': PromptEditor },
    data() {
        return {
            models: { local: [], huggingface: [] },
            selectedModel: '',
            loading: false,
            prompt: '',
            negativePrompt: '',
            params: {
                width: 512,
                height: 512,
                steps: 30,
                cfg_scale: 7.0
            },
            generating: false,
            results: [],
            modalImage: null,
            jobId: null
        };
    },
    computed: {
        canGenerate() {
            return this.prompt.trim() !== '' && !this.generating;
        }
    },
    async mounted() {
        await this.loadModels();
    },
    methods: {
        async loadModels() {
            try {
                const response = await axios.get(`${API_BASE}/models`);
                this.models = response.data.models;
            } catch (error) {
                console.error('Error loading models:', error);
            }
        },
        async loadModel() {
            if (!this.selectedModel) return;
            this.loading = true;
            try {
                await axios.post(`${API_BASE}/load-model`, { model_id: this.selectedModel });
                this.$emit('update-gpu');
            } catch (error) {
                console.error('Error loading model:', error);
                alert('Error loading model: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        async generate() {
            if (!this.canGenerate) return;
            
            this.generating = true;
            this.progress = 0;
            this.results = [];
            
            try {
                const response = await axios.post(`${API_BASE}/generate`, {
                    prompt: this.prompt,
                    negative_prompt: this.negativePrompt || undefined,
                    ...this.params
                });
                
                this.jobId = response.data.job_id;
                this.pollStatus();
            } catch (error) {
                console.error('Error generating:', error);
                alert('Error: ' + (error.response?.data?.detail || error.message));
                this.generating = false;
            }
        },
        async pollStatus() {
            const interval = setInterval(async () => {
                try {
                    const response = await axios.get(`${API_BASE}/status/${this.jobId}`);
                    const job = response.data.job;
                    
                    if (job.status === 'completed') {
                        this.results = job.images;
                        this.generating = false;
                        clearInterval(interval);
                        this.$emit('update-gpu');
                        utils.logGeneration({
                            type: 'simple',
                            prompt: this.prompt,
                            negative_prompt: this.negativePrompt,
                            params: this.params,
                            results: job.images
                        });
                    } else if (job.status === 'failed') {
                        alert('Generation failed: ' + job.error);
                        this.generating = false;
                        clearInterval(interval);
                    }
                } catch (error) {
                    console.error('Error polling:', error);
                }
            }, 2000);
        },
        getImageUrl(path) {
            return `${API_BASE}/${path}`;
        },
        showImage(img) {
            this.modalImage = img;
        }
    }
};

// ============================================================================
// TAB: BATCH (continuará...)
// ============================================================================

const BatchTab = {
    template: `
        <div class="space-y-4">
            <!-- Model Selection -->
            <div class="card">
                <h3 class="font-semibold mb-3">Model</h3>
                <div class="flex gap-2">
                    <select 
                        v-model="selectedModel" 
                        class="flex-1 px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text"
                        :disabled="loading"
                    >
                        <option value="">-- Select model --</option>
                        <optgroup label="Local Models">
                            <option v-for="model in models.local" :key="model.id" :value="model.id">
                                {{ model.name }}
                            </option>
                        </optgroup>
                        <optgroup label="HuggingFace Models">
                            <option v-for="model in models.huggingface" :key="model.id" :value="model.id">
                                {{ model.name }}
                            </option>
                        </optgroup>
                    </select>
                    <button 
                        @click="loadModel" 
                        :disabled="!selectedModel || loading"
                        class="btn btn-primary px-6"
                    >
                        {{ loading ? 'Loading...' : 'Load' }}
                    </button>
                </div>
            </div>

            <!-- Prompt Editor -->
            <prompt-editor
                v-model:prompt="prompt"
                v-model:negative-prompt="negativePrompt"
                :saved-prompts="savedPrompts"
                @save="$emit('save-prompt', $event)"
            ></prompt-editor>

            <!-- Parameters -->
            <div class="card">
                <h3 class="font-semibold mb-3">Parameters</h3>
                
                <div class="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label class="block text-sm mb-1">Width</label>
                        <input v-model.number="params.width" type="number" step="64" min="256" max="1024">
                        <p class="text-xs text-muted mt-1">Image width in pixels</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Height</label>
                        <input v-model.number="params.height" type="number" step="64" min="256" max="1024">
                        <p class="text-xs text-muted mt-1">Image height in pixels</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Steps</label>
                        <input v-model.number="params.steps" type="number" min="1" max="150">
                        <p class="text-xs text-muted mt-1">Number of denoising steps</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">CFG Scale</label>
                        <input v-model.number="params.cfg_scale" type="number" step="0.5" min="1" max="20">
                        <p class="text-xs text-muted mt-1">Classifier-free guidance scale</p>
                    </div>
                </div>

                <div class="mb-4">
                    <label class="block text-sm mb-1">Iterations</label>
                    <input v-model.number="params.iterations" type="number" min="1" max="50">
                    <p class="text-xs text-muted mt-1">Number of images to generate</p>
                </div>

                <div>
                    <label class="block text-sm mb-2">Random Seed Each Time</label>
                    <div class="flex gap-4">
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="radio" v-model="params.randomSeed" :value="true" name="randomSeed">
                            <span class="text-sm">Yes</span>
                        </label>
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="radio" v-model="params.randomSeed" :value="false" name="randomSeed">
                            <span class="text-sm">No</span>
                        </label>
                    </div>
                    <p class="text-xs text-muted mt-1">Generate with different seed each iteration</p>
                </div>
            </div>

            <!-- Generate Button -->
            <button 
                @click="generateBatch" 
                :disabled="!canGenerate"
                class="btn btn-primary w-full py-3 text-lg font-semibold"
            >
                {{ generating ? 'Generating Batch...' : 'GENERATE BATCH' }}
            </button>

            <!-- Results -->
            <div v-if="results.length > 0" class="card">
                <h3 class="font-semibold mb-3">Results ({{ results.length }} images)</h3>
                <div class="image-grid">
                    <div v-for="(img, i) in results" :key="i" class="image-item" @click="showImage(img)">
                        <img :src="getImageUrl(img)" class="w-full h-auto">
                    </div>
                </div>
            </div>

            <!-- Modal -->
            <div v-if="modalImage" class="modal" @click.self="modalImage = null">
                <div class="modal-content">
                    <button @click="modalImage = null" class="modal-close">&times;</button>
                    <img :src="getImageUrl(modalImage)" class="w-full h-auto">
                </div>
            </div>
        </div>
    `,
    props: ['savedPrompts'],
    emits: ['save-prompt', 'update-gpu'],
    components: {
        'prompt-editor': PromptEditor
    },
    data() {
        return {
            models: { local: [], huggingface: [] },
            selectedModel: '',
            loading: false,
            prompt: '',
            negativePrompt: '',
            params: {
                width: 512,
                height: 512,
                steps: 30,
                cfg_scale: 7.0,
                iterations: 2,
                randomSeed: true
            },
            generating: false,
            results: [],
            modalImage: null,
            jobId: null
        };
    },
    async mounted() {
        await this.loadModels();
    },
    computed: {
        canGenerate() {
            return this.prompt.trim() !== '' && !this.generating;
        }
    },
    methods: {
        async loadModels() {
            try {
                const response = await axios.get(`${API_BASE}/models`);
                this.models = response.data.models;
            } catch (error) {
                console.error('Error loading models:', error);
            }
        },
        async loadModel() {
            if (!this.selectedModel) return;
            
            this.loading = true;
            try {
                await axios.post(`${API_BASE}/load-model`, { model_id: this.selectedModel });
                this.$emit('update-gpu');
            } catch (error) {
                console.error('Error loading model:', error);
                alert('Error loading model: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        async generateBatch() {
            if (!this.canGenerate) return;
            
            this.generating = true;
            this.results = [];
            
            try {
                // Crear array de prompts (mismo prompt repetido)
                const prompts = Array(this.params.iterations).fill(this.prompt);
                
                const batchConfig = {
                    prompts: prompts,
                    negative_prompt: this.negativePrompt,
                    width: this.params.width,
                    height: this.params.height,
                    steps: this.params.steps,
                    cfg_scale: this.params.cfg_scale,
                    seeds: this.params.randomSeed ? null : undefined
                };
                
                const response = await axios.post(`${API_BASE}/batch/generate`, batchConfig);
                this.jobId = response.data.job_id;
                this.pollStatus();
            } catch (error) {
                console.error('Error generating batch:', error);
                alert('Error: ' + (error.response?.data?.detail || error.message));
                this.generating = false;
            }
        },
        async pollStatus() {
            const interval = setInterval(async () => {
                try {
                    const response = await axios.get(`${API_BASE}/status/${this.jobId}`);
                    const job = response.data.job;
                    
                    if (job.status === 'completed') {
                        this.results = job.images;
                        this.generating = false;
                        clearInterval(interval);
                        this.$emit('update-gpu');
                    } else if (job.status === 'failed') {
                        alert('Batch generation failed: ' + job.error);
                        this.generating = false;
                        clearInterval(interval);
                    }
                } catch (error) {
                    console.error('Error polling:', error);
                }
            }, 3000);
        },
        getImageUrl(path) {
            return `${API_BASE}/${path}`;
        },
        showImage(img) {
            this.modalImage = img;
        }
    }
};

const ControlNetTab = {
    template: `
        <div class="space-y-4">
            <!-- Model Selection -->
            <div class="card">
                <h3 class="font-semibold mb-3">Model</h3>
                <div class="flex gap-2">
                    <select 
                        v-model="selectedModel" 
                        class="flex-1 px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text"
                        :disabled="loading"
                    >
                        <option value="">-- Select model --</option>
                        <optgroup label="Local Models">
                            <option v-for="model in models.local" :key="model.id" :value="model.id">
                                {{ model.name }}
                            </option>
                        </optgroup>
                        <optgroup label="HuggingFace Models">
                            <option v-for="model in models.huggingface" :key="model.id" :value="model.id">
                                {{ model.name }}
                            </option>
                        </optgroup>
                    </select>
                    <button 
                        @click="loadModel" 
                        :disabled="!selectedModel || loading"
                        class="btn btn-primary px-6"
                    >
                        {{ loading ? 'Loading...' : 'Load' }}
                    </button>
                </div>
            </div>

            <!-- Prompt Editor -->
            <prompt-editor
                v-model:prompt="prompt"
                v-model:negative-prompt="negativePrompt"
                :saved-prompts="savedPrompts"
                @save="$emit('save-prompt', $event)"
            ></prompt-editor>

            <!-- Image Preview Grid -->
            <div class="card">
                <h3 class="font-semibold mb-3">Images</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; border: 1px solid #2a2a2a; border-radius: 4px; padding: 12px; background: #1a1a1a;">
                    <!-- Source Image -->
                    <div style="border-right: 1px solid #2a2a2a; padding-right: 12px; min-width: 0;">
                        <p class="text-xs text-muted mb-2 text-center font-semibold">Source Image</p>
                        <div style="height: 300px; width: 100%; position: relative;">
                            <div v-if="sourceImage" style="width: 100%; height: 100%; border: 1px solid #2a2a2a; border-radius: 4px; overflow: hidden; background: #0f0f0f; display: flex; align-items: center; justify-content: center; position: relative;">
                                <img :src="sourceImage" style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;">
                                <button @click="clearSource" style="position: absolute; top: 8px; right: 8px; width: 24px; height: 24px; background: rgba(0,0,0,0.8); color: white; border: none; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; line-height: 1; font-weight: bold;" title="Clear image">
                                    ×
                                </button>
                            </div>
                            <div v-else class="w-full h-full bg-dark-bg border border-dark-border rounded flex flex-col items-center justify-center gap-2 p-3">
                                <button @click="$refs.fileInput.click()" class="btn btn-secondary text-sm">
                                    Upload File
                                </button>
                                <span class="text-xs text-muted">or</span>
                                <button @click="openGalleryModal" class="btn btn-secondary text-sm">
                                    Choose from Gallery
                                </button>
                            </div>
                        </div>
                        <input 
                            ref="fileInput"
                            type="file" 
                            @change="handleImageUpload" 
                            accept="image/*"
                            style="display: none;"
                        >
                    </div>

                    <!-- Canny Edges -->
                    <div style="border-right: 1px solid #2a2a2a; padding-left: 12px; padding-right: 12px; min-width: 0;">
                        <p class="text-xs text-muted mb-2 text-center font-semibold">Canny Edges</p>
                        <div style="width: 100%; height: 300px; border: 1px solid #2a2a2a; border-radius: 4px; overflow: hidden; background: #0f0f0f; display: flex; align-items: center; justify-content: center;">
                            <img v-if="cannyImage" :src="cannyImage" style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;">
                            <span v-else class="text-xs text-muted text-center">Preview Canny first</span>
                        </div>
                    </div>

                    <!-- Result -->
                    <div style="padding-left: 12px; min-width: 0;">
                        <p class="text-xs text-muted mb-2 text-center font-semibold">Generated Result</p>
                        <div style="width: 100%; height: 300px; border: 1px solid #2a2a2a; border-radius: 4px; overflow: hidden; background: #0f0f0f; display: flex; align-items: center; justify-content: center; cursor: pointer;" @click="resultImage && showModal(resultImage)">
                            <img v-if="resultImage" :src="resultImage" style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;">
                            <span v-else class="text-xs text-muted text-center">Generate first</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Canny Parameters -->
            <div v-if="sourceImage" class="card">
                <h3 class="font-semibold mb-3">Canny Edge Detection</h3>
                <div class="grid grid-cols-2 gap-4 mb-3">
                    <div>
                        <label class="block text-sm mb-1">Low Threshold</label>
                        <input v-model.number="cannyParams.low" type="number" min="0" max="255">
                        <p class="text-xs text-muted mt-1">Lower bound for edge detection</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">High Threshold</label>
                        <input v-model.number="cannyParams.high" type="number" min="0" max="255">
                        <p class="text-xs text-muted mt-1">Upper bound for edge detection</p>
                    </div>
                </div>
                <button @click="previewCanny" :disabled="processingCanny" class="btn btn-secondary w-full">
                    {{ processingCanny ? 'Processing...' : 'Preview Canny Edges' }}
                </button>
            </div>

            <!-- Generation Parameters -->
            <div class="card">
                <h3 class="font-semibold mb-3">Parameters</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm mb-1">Width</label>
                        <input v-model.number="params.width" type="number" step="64" min="256" max="1024">
                        <p class="text-xs text-muted mt-1">Image width in pixels</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Height</label>
                        <input v-model.number="params.height" type="number" step="64" min="256" max="1024">
                        <p class="text-xs text-muted mt-1">Image height in pixels</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Steps</label>
                        <input v-model.number="params.steps" type="number" min="1" max="150">
                        <p class="text-xs text-muted mt-1">Number of denoising steps</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">CFG Scale</label>
                        <input v-model.number="params.cfg_scale" type="number" step="0.5" min="1" max="20">
                        <p class="text-xs text-muted mt-1">Classifier-free guidance scale</p>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">ControlNet Scale</label>
                        <input v-model.number="params.controlnet_scale" type="number" step="0.1" min="0" max="2">
                        <p class="text-xs text-muted mt-1">Strength of ControlNet guidance</p>
                    </div>
                </div>
            </div>

            <!-- Generate Button -->
            <button 
                @click="generate" 
                :disabled="!canGenerate"
                class="btn btn-primary w-full py-3 text-lg font-semibold"
            >
                {{ generating ? 'Generating...' : 'GENERATE' }}
            </button>

            <!-- Gallery Modal -->
            <div v-if="showGallery" class="modal" @click.self="showGallery = false">
                <div class="modal-content" style="max-width: 1200px;">
                    <button @click="showGallery = false" class="modal-close">&times;</button>
                    <h3 class="text-xl font-semibold mb-4">Choose from Gallery</h3>
                    <div class="image-grid" style="max-height: 600px; overflow-y: auto;">
                        <div v-for="img in galleryImages" :key="img.path" class="image-item cursor-pointer" @click="selectFromGallery(img)">
                            <img :src="getImageUrl(img.path)" class="w-full h-auto">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Result Modal -->
            <div v-if="modalImage" class="modal" @click.self="modalImage = null">
                <div class="modal-content">
                    <button @click="modalImage = null" class="modal-close">&times;</button>
                    <img :src="modalImage" class="w-full h-auto">
                </div>
            </div>
        </div>
    `,
    props: ['savedPrompts'],
    emits: ['save-prompt', 'update-gpu'],
    components: {
        'prompt-editor': PromptEditor
    },
    data() {
        return {
            models: { local: [], huggingface: [] },
            selectedModel: '',
            loading: false,
            sourceImage: null,
            cannyImage: null,
            resultImage: null,
            cannyParams: {
                low: 100,
                high: 200
            },
            processingCanny: false,
            prompt: '',
            negativePrompt: '',
            params: {
                width: 512,
                height: 512,
                steps: 30,
                cfg_scale: 7.5,
                controlnet_scale: 1.0
            },
            generating: false,
            modalImage: null,
            uploadedFile: null,
            selectedGalleryPath: null,
            showGallery: false,
            galleryImages: []
        };
    },
    async mounted() {
        await this.loadModels();
    },
    computed: {
        canGenerate() {
            return this.prompt.trim() !== '' && this.sourceImage && !this.generating;
        }
    },
    methods: {
        async loadModels() {
            try {
                const response = await axios.get(`${API_BASE}/models`);
                this.models = response.data.models;
            } catch (error) {
                console.error('Error loading models:', error);
            }
        },
        async loadModel() {
            if (!this.selectedModel) return;
            
            this.loading = true;
            try {
                await axios.post(`${API_BASE}/load-model`, { model_id: this.selectedModel });
                this.$emit('update-gpu');
            } catch (error) {
                console.error('Error loading model:', error);
                alert('Error loading model: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            this.uploadedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                this.sourceImage = e.target.result;
                this.cannyImage = null;
                this.resultImage = null;
            };
            reader.readAsDataURL(file);
        },
        clearSource() {
            this.sourceImage = null;
            this.cannyImage = null;
            this.resultImage = null;
            this.uploadedFile = null;
            this.selectedGalleryPath = null;
        },
        async openGalleryModal() {
            try {
                const response = await axios.get(`${API_BASE}/gallery`);
                this.galleryImages = response.data.images;
                this.showGallery = true;
            } catch (error) {
                console.error('Error loading gallery:', error);
                alert('Error loading gallery');
            }
        },
        async selectFromGallery(img) {
            try {
                // Usar URL relativa para evitar problemas de CORS
                const imagePath = `/${img.path}`;
                const response = await fetch(imagePath);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const blob = await response.blob();
                
                // Crear File desde el blob
                this.uploadedFile = new File([blob], img.filename || 'image.png', { type: blob.type || 'image/png' });
                this.sourceImage = imagePath;
                this.selectedGalleryPath = null;
                this.cannyImage = null;
                this.resultImage = null;
                this.showGallery = false;
            } catch (error) {
                console.error('Error loading image from gallery:', error);
                alert('Error loading image from gallery: ' + error.message);
            }
        },
        async previewCanny() {
            if (!this.uploadedFile) return;
            
            this.processingCanny = true;
            try {
                const formData = new FormData();
                formData.append('file', this.uploadedFile);
                formData.append('controlnet_type', 'canny');
                formData.append('width', this.params.width);
                formData.append('height', this.params.height);
                formData.append('low_threshold', this.cannyParams.low);
                formData.append('high_threshold', this.cannyParams.high);
                
                const response = await axios.post(`${API_BASE}/controlnet/preprocess`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                
                // Limpiar primero para forzar re-render
                this.cannyImage = null;
                // Usar nextTick para asegurar que Vue detecte el cambio
                await this.$nextTick();
                this.cannyImage = response.data.processed_image;
            } catch (error) {
                console.error('Error processing Canny:', error);
                console.error('Error details:', error.response?.data);
                const errorMsg = error.response?.data?.detail || JSON.stringify(error.response?.data) || error.message;
                alert('Error processing Canny: ' + errorMsg);
            } finally {
                this.processingCanny = false;
            }
        },
        async generate() {
            if (!this.canGenerate) return;
            
            this.generating = true;
            try {
                // Cargar ControlNet si no está cargado
                try {
                    await axios.post(`${API_BASE}/controlnet/load`, {
                        controlnet_type: 'canny'
                    });
                } catch (loadError) {
                    // Si ya está cargado, continuar
                    if (!loadError.response?.data?.detail?.includes('ya está cargado')) {
                        console.warn('ControlNet load warning:', loadError);
                    }
                }
                
                const formData = new FormData();
                formData.append('file', this.uploadedFile);
                formData.append('prompt', this.prompt);
                formData.append('negative_prompt', this.negativePrompt || '');
                formData.append('width', this.params.width);
                formData.append('height', this.params.height);
                formData.append('steps', this.params.steps);
                formData.append('cfg_scale', this.params.cfg_scale);
                formData.append('controlnet_scale', this.params.controlnet_scale);
                formData.append('controlnet_type', 'canny');
                
                const response = await axios.post(`${API_BASE}/controlnet/generate`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                
                this.resultImage = `${API_BASE}/${response.data.image_path}`;
                this.$emit('update-gpu');
            } catch (error) {
                console.error('Error generating:', error);
                console.error('Error details:', error.response?.data);
                const errorMsg = error.response?.data?.detail || JSON.stringify(error.response?.data) || error.message;
                alert('Error generating: ' + errorMsg);
            } finally {
                this.generating = false;
            }
        },
        showModal(img) {
            this.modalImage = img;
        },
        getImageUrl(path) {
            return `${API_BASE}/${path}`;
        }
    }
};

const GalleryTab = {
    template: `
        <div class="space-y-4">
            <!-- Filters -->
            <div class="card">
                <div class="grid grid-cols-4 gap-4">
                    <div>
                        <label class="block text-sm mb-1">Sort By</label>
                        <select v-model="sortBy" @change="loadGallery" class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text">
                            <option value="date_desc">Newest First</option>
                            <option value="date_asc">Oldest First</option>
                            <option value="name_asc">Name A-Z</option>
                            <option value="name_desc">Name Z-A</option>
                        </select>
                    </div>
                    <div class="col-span-2">
                        <label class="block text-sm mb-1">Search</label>
                        <input 
                            v-model="searchQuery" 
                            @input="filterImages"
                            type="text" 
                            class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-dark-text" 
                            placeholder="Search by filename..."
                        >
                    </div>
                    <div class="flex items-end">
                        <button @click="loadGallery" class="btn btn-secondary w-full">Refresh</button>
                    </div>
                </div>
            </div>

            <!-- Loading -->
            <div v-if="loading" class="card text-center py-8 text-muted">
                Loading gallery...
            </div>

            <!-- Empty -->
            <div v-else-if="filteredImages.length === 0" class="card text-center py-8 text-muted">
                No images found
            </div>

            <!-- Image Grid -->
            <div v-else class="card">
                <h3 class="font-semibold mb-3">{{ filteredImages.length }} images</h3>
                <div class="image-grid">
                    <div v-for="(img, i) in filteredImages" :key="img.path" class="image-item" @click="openModal(i)">
                        <img :src="getImageUrl(img.path)" class="w-full h-auto">
                    </div>
                </div>
            </div>

            <!-- Modal -->
            <div v-if="modalIndex !== null" class="modal" @click.self="closeModal">
                <div class="modal-content">
                    <button @click="closeModal" class="modal-close">&times;</button>
                    <img :src="getImageUrl(currentImage.path)" class="w-full h-auto">
                    <div class="modal-footer">
                        <button @click="prevImage" class="btn btn-secondary" :disabled="modalIndex === 0">&lt; Prev</button>
                        <button @click="deleteImage" class="btn btn-danger">Delete</button>
                        <button @click="nextImage" class="btn btn-secondary" :disabled="modalIndex === filteredImages.length - 1">Next &gt;</button>
                    </div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            images: [],
            filteredImages: [],
            sortBy: 'date_desc',
            searchQuery: '',
            loading: false,
            modalIndex: null
        };
    },
    computed: {
        currentImage() {
            return this.modalIndex !== null ? this.filteredImages[this.modalIndex] : null;
        }
    },
    async mounted() {
        await this.loadGallery();
    },
    methods: {
        async loadGallery() {
            this.loading = true;
            try {
                const response = await axios.get(`${API_BASE}/gallery?sort=${this.sortBy}`);
                this.images = response.data.images || [];
                this.filterImages();
            } catch (error) {
                console.error('Error loading gallery:', error);
            } finally {
                this.loading = false;
            }
        },
        filterImages() {
            if (!this.searchQuery.trim()) {
                this.filteredImages = this.images;
            } else {
                const query = this.searchQuery.toLowerCase();
                this.filteredImages = this.images.filter(img => 
                    img.name.toLowerCase().includes(query)
                );
            }
        },
        openModal(index) {
            this.modalIndex = index;
        },
        closeModal() {
            this.modalIndex = null;
        },
        prevImage() {
            if (this.modalIndex > 0) this.modalIndex--;
        },
        nextImage() {
            if (this.modalIndex < this.filteredImages.length - 1) this.modalIndex++;
        },
        async deleteImage() {
            if (!confirm('Delete this image from disk?')) return;
            
            try {
                await axios.delete(`${API_BASE}/gallery/${encodeURIComponent(this.currentImage.path)}`);
                this.images = this.images.filter(img => img.path !== this.currentImage.path);
                this.filterImages();
                
                if (this.filteredImages.length === 0) {
                    this.closeModal();
                } else if (this.modalIndex >= this.filteredImages.length) {
                    this.modalIndex = this.filteredImages.length - 1;
                }
            } catch (error) {
                console.error('Error deleting image:', error);
                alert('Error deleting image');
            }
        },
        getImageUrl(path) {
            return `${API_BASE}/${path}`;
        }
    }
};

// ============================================================================
// APP PRINCIPAL
// ============================================================================

createApp({
    components: {
        'generate-tab': GenerateTab,
        'batch-tab': BatchTab,
        'controlnet-tab': ControlNetTab,
        'gallery-tab': GalleryTab
    },
    data() {
        return {
            activeTab: 'generate',
            tabs: [
                { id: 'generate', name: 'GENERATE' },
                { id: 'batch', name: 'BATCH' },
                { id: 'controlnet', name: 'CONTROLNET' },
                { id: 'gallery', name: 'GALLERY' }
            ],
            savedPrompts: [],
            gpuStats: { allocated_gb: 0, total_gb: 0, utilization: null, temperature: null },
            currentModel: null
        };
    },
    async mounted() {
        await this.loadSavedPrompts();
        await this.updateGPUStats();
        await this.updateCurrentModel();
        setInterval(() => this.updateGPUStats(), 5000);
    },
    computed: {
        tempClass() {
            if (!this.gpuStats.temperature) return '';
            if (this.gpuStats.temperature < 65) return 'temp-normal';
            if (this.gpuStats.temperature < 75) return 'temp-warm';
            return 'temp-hot';
        }
    },
    methods: {
        async loadSavedPrompts() {
            try {
                const response = await axios.get(`${API_BASE}/prompts`);
                this.savedPrompts = response.data.prompts || [];
            } catch (error) {
                console.error('Error loading prompts:', error);
            }
        },
        async savePrompt(prompt) {
            try {
                await axios.post(`${API_BASE}/prompts`, prompt);
                await this.loadSavedPrompts();
            } catch (error) {
                console.error('Error saving prompt:', error);
                alert('Error saving prompt');
            }
        },
        async updateGPUStats() {
            try {
                const response = await axios.get(`${API_BASE}/gpu-info`);
                const data = response.data;
                
                // Mapear nombres de campos del backend al frontend
                this.gpuStats = {
                    allocated_gb: data.allocated_memory_gb || 0,
                    total_gb: data.total_memory_gb || 0,
                    utilization: data.gpu_utilization_percent,
                    temperature: data.temperature_celsius
                };
            } catch (error) {
                console.error('Error updating GPU stats:', error);
            }
        },
        async updateCurrentModel() {
            try {
                const response = await axios.get(`${API_BASE}/models`);
                this.currentModel = response.data.current_model?.model_name || 'None';
            } catch (error) {
                console.error('Error updating current model:', error);
            }
        },
        async updateAll() {
            await this.updateGPUStats();
            await this.updateCurrentModel();
        }
    }
}).mount('#app');
