# Florence-2 Playground — Detailed Project Plan

**Project:** Florence-2 Vision Playground  
**Goal:** A polished, full-stack web application for interactive exploration of Microsoft Florence-2 vision-language model capabilities  
**Stack:** React (Vite + Tailwind + shadcn/ui) + FastAPI + HuggingFace Transformers  
**Created:** 2026-05-11

---

## 1. Project Structure

```
Playground/
├── PLAN.md                     ← This file
├── backend/
│   ├── main.py                 ← FastAPI app entry point
│   ├── model.py                ← Florence-2 model loading & inference
│   ├── tasks.py                ← Task definitions and routing
│   ├── visualizer.py           ← Bbox / mask / OCR overlay rendering
│   ├── schemas.py              ← Pydantic request/response models
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── layout/
│       │   │   ├── Sidebar.tsx         ← Task category navigation
│       │   │   ├── Header.tsx          ← App title + theme toggle
│       │   │   └── HistoryDrawer.tsx   ← Session history panel
│       │   ├── upload/
│       │   │   ├── ImageDropzone.tsx   ← Drag & drop upload
│       │   │   └── ImagePreview.tsx    ← Preview with crop/zoom
│       │   ├── tasks/
│       │   │   ├── TaskPanel.tsx       ← Task selector + params
│       │   │   ├── TaskCard.tsx        ← Individual task card with description
│       │   │   └── TextInput.tsx       ← Optional text prompt input
│       │   ├── results/
│       │   │   ├── ResultPanel.tsx     ← Annotated image + JSON output
│       │   │   ├── AnnotatedImage.tsx  ← Canvas overlay for bboxes/masks
│       │   │   ├── JsonViewer.tsx      ← Collapsible structured JSON
│       │   │   └── ExportButton.tsx    ← Download PNG + JSON
│       │   └── ui/                     ← shadcn/ui components
│       ├── hooks/
│       │   ├── useInference.ts         ← API call + loading state
│       │   └── useHistory.ts           ← Session history management
│       ├── lib/
│       │   ├── api.ts                  ← Axios API client
│       │   └── utils.ts
│       └── types/
│           └── index.ts                ← Shared TypeScript types
└── docker-compose.yml          ← Optional: containerized deployment
```

---

## 2. Florence-2 Tasks Supported

### Group A — Captioning
| Task Key | Prompt | Description |
|---|---|---|
| Caption | `<CAPTION>` | Short one-line description of the image |
| Detailed Caption | `<DETAILED_CAPTION>` | Paragraph-level description |
| More Detailed Caption | `<MORE_DETAILED_CAPTION>` | Rich, exhaustive description |

### Group B — Object Detection & Localization
| Task Key | Prompt | Description |
|---|---|---|
| Object Detection | `<OD>` | Detect objects with bounding boxes |
| Dense Region Caption | `<DENSE_REGION_CAPTION>` | Caption for every detected region |
| Region Proposal | `<REGION_PROPOSAL>` | Propose candidate regions |
| Open Vocabulary Detection | `<OPEN_VOCABULARY_DETECTION>` | Detect user-specified objects |

### Group C — Grounding
| Task Key | Prompt | Description |
|---|---|---|
| Caption to Phrase Grounding | `<CAPTION_TO_PHRASE_GROUNDING>` | Ground phrases in a caption to image regions |
| Region to Category | `<REGION_TO_CATEGORY>` | Classify a user-drawn region |
| Region to Description | `<REGION_TO_DESCRIPTION>` | Describe a user-drawn region |

### Group D — Segmentation
| Task Key | Prompt | Description |
|---|---|---|
| Referring Expression Segmentation | `<REFERRING_EXPRESSION_SEGMENTATION>` | Segment object described in text |
| Region to Segmentation | `<REGION_TO_SEGMENTATION>` | Segment within a bounding box |

### Group E — OCR
| Task Key | Prompt | Description |
|---|---|---|
| OCR | `<OCR>` | Extract all text from image |
| OCR with Region | `<OCR_WITH_REGION>` | Extract text with bounding box locations |

### Group F — Cascaded (Multi-step)
| Task Key | Description |
|---|---|
| Caption + Grounding | Run Caption, then ground the output phrases |
| Detailed Caption + Grounding | Run Detailed Caption, then ground phrases |
| More Detailed Caption + Grounding | Run More Detailed Caption, then ground phrases |

---

## 3. Backend Design

### 3.1 FastAPI Endpoints

```
POST   /api/infer          ← Main inference endpoint
GET    /api/tasks          ← Returns task list with metadata
GET    /api/models         ← Returns available model variants
GET    /api/health         ← Health check + GPU status
```

### 3.2 Inference Request Schema
```json
{
  "task": "<OD>",
  "model_id": "microsoft/Florence-2-large",
  "text_input": "optional prompt text",
  "image": "base64-encoded image string"
}
```

### 3.3 Inference Response Schema
```json
{
  "task": "<OD>",
  "raw_output": { ... },
  "annotated_image": "base64-encoded PNG",
  "processing_time_ms": 340
}
```

### 3.4 Model Loading Strategy
- Load model once at startup into GPU memory
- Default: `microsoft/Florence-2-large` (best quality)
- Allow switching model at runtime via request param
- Use `torch.float16` for memory efficiency on GPU

### 3.5 Key Python Dependencies
```
torch>=2.0.0
torchvision
transformers>=4.38.0
timm
einops
fastapi
uvicorn[standard]
python-multipart
pillow
matplotlib
numpy
python-dotenv
```

---

## 4. Frontend Design

### 4.1 Theme & Styling
- **Dark mode by default**, with a toggle for light mode
- Color palette: Deep navy background (#0F172A), accent purple/violet (#7C3AED)
- Typography: Inter (body), JetBrains Mono (code/JSON output)
- Fully responsive: works on desktop and tablet

### 4.2 Layout (Desktop)
```
┌─────────────────────────────────────────────────────────────┐
│  Header: "Florence-2 Playground"           [Light/Dark] [?] │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│   Sidebar    │          Main Content Area                   │
│              │                                              │
│  📷 Caption  │  ┌─────────────┐   ┌──────────────────────┐ │
│  🔍 Detect   │  │ Image Upload│   │   Result Panel       │ │
│  ✏️ OCR      │  │             │   │                      │ │
│  🎭 Segment  │  │  Drop here  │   │  [Annotated Image]   │ │
│  📍 Ground   │  │  or click   │   │  [JSON Output]       │ │
│  ⛓️ Cascade  │  └─────────────┘   │  [Export Button]     │ │
│              │                   └──────────────────────┘ │
│  ──────────  │  ┌─────────────────────────────────────┐   │
│  History     │  │  Task Panel                         │   │
│  [thumb] OD  │  │  [Task cards with icons+description]│   │
│  [thumb] OCR │  │  [Text input if needed]             │   │
│  [thumb] Cap │  │  [Run Button]                       │   │
│              │  └─────────────────────────────────────┘   │
└──────────────┴──────────────────────────────────────────────┘
```

### 4.3 Key UI/UX Features
1. **Drag & Drop Upload** — highlight on drag-over, show thumbnail preview
2. **Task Cards** — each task has icon, name, short description, and example use case
3. **Animated Results** — bounding boxes animate in (draw progressively)
4. **Mask Opacity Slider** — for segmentation results, user can adjust overlay opacity
5. **Interactive JSON** — collapsible tree view for raw model output
6. **One-click Export** — downloads annotated image as PNG + results as JSON
7. **Session History** — last 10 runs stored in browser (localStorage), shown as thumbnails
8. **Loading Skeleton** — smooth skeleton UI while inference runs
9. **Error Toast** — friendly error messages if model fails
10. **Keyboard Shortcut** — `Ctrl+Enter` to run inference

### 4.4 Key Frontend Dependencies
```json
{
  "react": "^18",
  "typescript": "^5",
  "vite": "^5",
  "tailwindcss": "^3",
  "@radix-ui/react-*": "latest",    ← shadcn/ui primitives
  "axios": "^1",
  "react-dropzone": "^14",
  "framer-motion": "^11",           ← animations
  "react-json-view-lite": "^1",     ← JSON viewer
  "lucide-react": "^0.400"          ← icons
}
```

---

## 5. Build & Run

### 5.1 Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2 Frontend
```bash
cd frontend
npm install
npm run dev        # development (port 5173)
npm run build      # production build → dist/
```

### 5.3 Production (serve together)
- FastAPI serves the React `dist/` as static files at `/`
- Single server on port 8000

---

## 6. Implementation Phases

### Phase 1 — Backend Core (Day 1–2)
- [ ] Set up FastAPI project skeleton
- [ ] Implement model loading (`model.py`)
- [ ] Implement all task prompts (`tasks.py`)
- [ ] Implement visualizer for bbox/mask/OCR (`visualizer.py`)
- [ ] Wire up `/api/infer` endpoint
- [ ] Test all 17 tasks with curl/Postman

### Phase 2 — Frontend Shell (Day 2–3)
- [ ] Scaffold React + Vite + Tailwind + shadcn/ui
- [ ] Build layout: Header, Sidebar, main grid
- [ ] Build ImageDropzone with preview
- [ ] Build TaskPanel with task cards
- [ ] Wire up API call via `useInference` hook

### Phase 3 — Result Visualization (Day 3–4)
- [ ] AnnotatedImage canvas component (bboxes + masks)
- [ ] JsonViewer component
- [ ] ExportButton (PNG + JSON download)
- [ ] Mask opacity slider for segmentation

### Phase 4 — Polish (Day 4–5)
- [ ] Session history with localStorage
- [ ] Framer Motion animations (bbox draw-in, panel transitions)
- [ ] Dark/light theme toggle
- [ ] Loading skeletons
- [ ] Error handling & toast notifications
- [ ] Keyboard shortcuts

### Phase 5 — Deploy & Demo (Day 5–6)
- [ ] Serve React build from FastAPI
- [ ] Configure for GPU server
- [ ] Final testing of all task groups
- [ ] Demo walkthrough preparation

---

## 7. Storage & Hardware Requirements

| Component | Requirement |
|---|---|
| Florence-2-large weights | ~1.5 GB |
| PyTorch + CUDA | ~3–5 GB |
| Python dependencies | ~500 MB |
| Node modules + build | ~300 MB |
| **Total (fresh)** | **~6–7 GB** |
| **Total (PyTorch pre-installed)** | **~2.5 GB** |

**GPU:** Minimum 8 GB VRAM recommended for Florence-2-large (float16)  
**RAM:** 16 GB system RAM recommended

---

## 8. Future Enhancements (Post-Capstone)
- Batch image processing (upload folder)
- Video frame-by-frame analysis
- Side-by-side model comparison (base vs large vs fine-tuned)
- User annotations — draw bounding boxes manually as input for grounding tasks
- API key–protected public deployment
