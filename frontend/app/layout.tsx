import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Sia Chat QnA",
  description: "Let me read the textbook and you can chat. I can assist you in learning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full`}>
        <div
          className="flex flex-col h-full md:p-8"
          style={{ background: "#f8f8f8"}}
        >
          {children}
        </div>
      </body>
    </html>
  );
}
