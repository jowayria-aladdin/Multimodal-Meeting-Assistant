import type { Metadata } from "next";
import { Inter } from "next/font/google";
// @ts-expect-error: CSS module declaration missing for globals.css
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter", 
  display: "swap",
});

export const metadata: Metadata = {
  title: "LughaCap | AI-Powered Multimodal Assistant",
  description: "LughaCap; for the words you speak, and the signs you show.",
  icons: {
    icon: "/LughaCap_Icon.png",
    apple: "/LughaCap_Icon.png", 
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased bg-white text-slate-900 flex flex-col min-h-screen`}>
        {children}
      </body>
    </html>
  );
}