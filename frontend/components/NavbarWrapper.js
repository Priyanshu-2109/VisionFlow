"use client"; // Mark this as a client component

import { usePathname } from "next/navigation";
import Navbar from "@/components/Navbar";

export default function NavbarWrapper() {
  const pathname = usePathname();

  // Hide Navbar on the login page
  if (pathname ==="/dashboard") return null;

  return <Navbar />;
}
