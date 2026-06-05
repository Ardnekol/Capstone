import { useCallback, useEffect, useState } from 'react';
import { Header } from '@/components/layout/Header';
import { HistoryDrawer } from '@/components/layout/HistoryDrawer';
import { Sidebar } from '@/components/layout/Sidebar';
import { ResultPanel } from '@/components/results/ResultPanel';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { ImageDropzone } from '@/components/upload/ImageDropzone';
import { TaskPanel } from '@/components/tasks/TaskPanel';
import { useHistory } from '@/hooks/useHistory';
import { useInference } from '@/hooks/useInference';
import { fetchHealth, fetchModels, fetchTasks } from '@/lib/api';
import { imageFileToThumbnail, formatMs } from '@/lib/utils';
import { useToast } from '@/components/ui/Toast';
import type { HealthStatus, HistoryItem, TaskGroupId, TaskInfo } from '@/types';

const DEFAULT_GROUP: TaskGroupId = 'caption';
const DEFAULT_TASK = '<CAPTION>';

export default function App() {
  // Theme
  const [isDark, setIsDark] = useState(true);

  // API data
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  // Selection state
  const [selectedGroup, setSelectedGroup] = useState<TaskGroupId>(DEFAULT_GROUP);
  const [selectedTask, setSelectedTask] = useState<string>(DEFAULT_TASK);
  const [modelId, setModelId] = useState('microsoft/Florence-2-large');

  // Image state
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  // Text / region input
  const [textInput, setTextInput] = useState('');

  // UI state
  const [historyOpen, setHistoryOpen] = useState(false);

  const { result, loading, error, run, reset } = useInference({
    image: file,
    task: selectedTask,
    modelId,
    textInput,
  });
  const { history, push: pushHistory, clear: clearHistory } = useHistory();
  const { toast } = useToast();

  // ── Bootstrap ─────────────────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([fetchTasks(), fetchModels(), fetchHealth()])
      .then(([t, m, h]) => {
        setTasks(t);
        setModels(m.models);
        setModelId(m.default);
        setHealth(h);
      })
      .catch(console.error);
  }, []);

  // Poll health every 30s
  useEffect(() => {
    const id = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => {});
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  // Show error toast whenever inference fails
  useEffect(() => {
    if (error) {
      toast({ type: 'error', title: 'Inference failed', message: error });
    }
  }, [error, toast]);

  // ── Theme ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
  }, [isDark]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleFile = useCallback(
    (f: File) => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      setFile(f);
      setImageUrl(URL.createObjectURL(f));
      reset();
    },
    [imageUrl, reset],
  );

  const handleClear = useCallback(() => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setFile(null);
    setImageUrl(null);
    reset();
  }, [imageUrl, reset]);

  const handleTaskSelect = useCallback(
    (key: string) => {
      setSelectedTask(key);
      setTextInput('');
      reset();
    },
    [reset],
  );

  const handleGroupSelect = useCallback(
    (g: TaskGroupId) => {
      setSelectedGroup(g);
      const first = tasks.find((t) => t.group === g);
      if (first) handleTaskSelect(first.key);
    },
    [tasks, handleTaskSelect],
  );

  const handleRun = useCallback(async () => {
    const res = await run();
    if (res && file) {
      const thumbnail = await imageFileToThumbnail(file);
      const item: HistoryItem = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        task_key: selectedTask,
        task_name: res.task_name,
        group: selectedGroup,
        thumbnail,
        result: res,
      };
      pushHistory(item);
      toast({
        type: 'success',
        title: `${res.task_name} complete`,
        message: `Finished in ${formatMs(res.processing_time_ms)}`,
      });
    }
  }, [run, file, selectedTask, selectedGroup, pushHistory, toast]);

  // ── Keyboard shortcut: Cmd/Ctrl + Enter ───────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (file && selectedTask && !loading) handleRun();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [file, selectedTask, loading, handleRun]);

  const currentTaskInfo = tasks.find((t) => t.key === selectedTask);
  const groupTasks = tasks.filter((t) => t.group === selectedGroup);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans">
      <ProgressBar loading={loading} />
      <Header
        isDark={isDark}
        onToggleTheme={() => setIsDark((d) => !d)}
        onOpenHistory={() => setHistoryOpen(true)}
        health={health}
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar selectedGroup={selectedGroup} onSelectGroup={handleGroupSelect} />

        {/* Main content */}
        <main className="flex-1 flex gap-4 p-4 overflow-auto">
          {/* Left column — upload + task */}
          <div className="w-[400px] shrink-0 flex flex-col gap-4">
            <ImageDropzone
              imageUrl={imageUrl}
              onFile={handleFile}
              onClear={handleClear}
            />
            <TaskPanel
              tasks={groupTasks}
              selectedTask={selectedTask}
              onSelectTask={handleTaskSelect}
              taskInfo={currentTaskInfo}
              textInput={textInput}
              onTextInput={setTextInput}
              models={models}
              modelId={modelId}
              onModelChange={setModelId}
              onRun={handleRun}
              loading={loading}
              canRun={!!file && !!selectedTask}
            />
          </div>

          {/* Right column — results */}
          <div className="flex-1 min-w-0">
            <ResultPanel
              result={result}
              loading={loading}
              error={error}
              originalImage={imageUrl}
            />
          </div>
        </main>
      </div>

      <HistoryDrawer
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        history={history}
        onClear={clearHistory}
        onSelect={(_item) => {
          setHistoryOpen(false);
        }}
      />
    </div>
  );
}
