import { useEffect, useState } from 'react';
import type { HistoryItem } from '@/types';

const KEY = 'f2-playground-history';
const MAX = 10;

export function useHistory() {
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(KEY) ?? '[]') as HistoryItem[];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    // Strip annotated_image (large base64 PNG) before persisting — thumbnails
    // are stored separately and are small enough. Full images can be 2-5 MB each,
    // which quickly exhausts localStorage's ~5 MB quota.
    const slim = history.map((item) => ({
      ...item,
      result: { ...item.result, annotated_image: null },
    }));
    try {
      localStorage.setItem(KEY, JSON.stringify(slim));
    } catch {
      // Still over quota after stripping images — trim to last 3 and retry
      try {
        localStorage.setItem(KEY, JSON.stringify(slim.slice(0, 3)));
      } catch {
        localStorage.removeItem(KEY);
      }
    }
  }, [history]);

  function push(item: HistoryItem) {
    setHistory((prev) => [item, ...prev].slice(0, MAX));
  }

  function clear() {
    setHistory([]);
    localStorage.removeItem(KEY);
  }

  return { history, push, clear };
}
