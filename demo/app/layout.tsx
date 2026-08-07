import type { Metadata } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.startsWith("localhost") ? "http" : "https");
  const metadataBase = new URL(`${protocol}://${host}`);

  return {
    metadataBase,
    title: "MAVIS · Memory-Aware Visual Inference",
    description: "Selective factory CCTV reasoning powered by evaluated EverOS memory.",
    openGraph: {
      title: "MAVIS · Memory-Aware Visual Inference",
      description: "See less. Understand what matters.",
      type: "website",
      images: [{ url: "/og.png", width: 1792, height: 1024, alt: "MAVIS selective CCTV evidence workflow" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "MAVIS · Memory-Aware Visual Inference",
      description: "See less. Understand what matters.",
      images: ["/og.png"],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
