"use client";
import Link from "next/link";
import Login from "./Login";
import { useState } from "react";

export default function Navbar() {
  const [login, setLogin] = useState(false);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      // Get the element's position relative to the viewport
      const elementPosition = element.getBoundingClientRect().top;
      // Get the current scroll position
      const offsetPosition = elementPosition + window.pageYOffset - 100; // Subtract 100px to account for navbar height

      // Scroll to the adjusted position smoothly
      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth",
      });
    }
  };

  // Map of navigation items to their corresponding section IDs
  const navItems = [
    { name: "Home", id: "hero" },
    { name: "Why Us?", id: "why-us" },
    { name: "Services", id: "services" },
    { name: "Effects", id: "effects" },
    { name: "AI", id: "ai" },
  ];

  return (
    <>
      <nav className="sticky top-0 w-full bg-[rgb(10,10,10)] text-white flex items-center justify-between px-12 h-20 border-b border-gray-700 z-50 transition duration-300 ease-in-out backdrop-blur-lg bg-opacity-70">
        {/* Left Brand Name */}
        <div className="text-2xl font-bold tracking-wide">VisionFlow</div>

        {/* Middle Links - Centered */}
        <div className="absolute left-1/2 transform -translate-x-1/2 flex space-x-8 text-lg font-medium">
          {navItems.map((item) => (
            <button
              key={item.name}
              onClick={() => scrollToSection(item.id)}
              className="hover:text-gray-400 transition cursor-pointer"
            >
              {item.name}
            </button>
          ))}
        </div>

        {/* Right Buttons */}
        <div className="flex space-x-6 text-lg items-center font-medium">
          <button
            onClick={() => scrollToSection("contact")}
            className="hover:text-gray-400 transition"
          >
            Contact
          </button>
          <Link
            href="/dashboard"
            className="bg-white text-black px-6 py-2 rounded-md"
          >
            Dashboard
          </Link>
          <button
            onClick={() => setLogin(true)}
            className="bg-white text-black px-6 py-2 rounded-md"
          >
            Log In
          </button>
        </div>
      </nav>
      {login && <Login onClose={() => setLogin(false)} />}
    </>
  );
}
