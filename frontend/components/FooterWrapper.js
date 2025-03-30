"use client"; // Mark this as a client component

import { usePathname } from "next/navigation";
import Footer from "@/components/Footer";

export default function NavbarWrapper() {
  const pathname = usePathname();

  // Hide Navbar 
  if (pathname === "/login" ||pathname === "/signup"  || pathname ==="/dashboard") return null;

  return <Footer />;
}
