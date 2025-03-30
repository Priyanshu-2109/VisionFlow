"use client";
import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="sticky top-0 w-full bg-[rgb(10,10,10)] text-white flex items-center justify-between px-12 h-20 border-b border-gray-700 z-50 transition duration-300 ease-in-out backdrop-blur-lg bg-opacity-70">
      {/* Left Brand Name */}
      <div className="text-2xl font-bold tracking-wide">VisionFlow</div>

      {/* Middle Links */}
      <div className="flex space-x-8 text-lg font-medium">
        {["Product", "Pricing", "Customers", "Blog", "Docs", "Changelog", "Company"].map((item) => (
          <Link key={item} href="#" className="hover:text-gray-400 transition">
            {item}
          </Link>
        ))}
      </div>

      {/* Right Buttons */}
      <div className="flex space-x-6 text-lg items-center font-medium">
        <Link href="#" className="hover:text-gray-400 transition">
          Contact
        </Link>
        <Link href="/dashboard" className="bg-white text-black px-6 py-2 rounded-md">
          Dashboard
        </Link>
        <Link href="/login" className="bg-white text-black px-6 py-2 rounded-md">
          Log In
        </Link>
      </div>
    </nav>
  );
}