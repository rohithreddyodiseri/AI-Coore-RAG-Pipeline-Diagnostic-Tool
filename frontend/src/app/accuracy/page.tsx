import type { Metadata } from "next";
import AccuracyPage from "@/features/accuracy/AccuracyPage";

export const metadata: Metadata = { title: "Diagnostic Accuracy | RAG Diagnostic Lab" };

export default function Page() {
  return <AccuracyPage />;
}
