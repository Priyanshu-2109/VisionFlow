"use client";
import Link from "next/link";
import Login from "./Login";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "react-toastify";

export default function Navbar() {
  const [login, setLogin] = useState(false);
  const [user, setUser] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [menuVisible, setMenuVisible] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const userData = localStorage.getItem("user");
    if (userData) {
      setUser(JSON.parse(userData));
    }
  }, []);

  // Handle smooth menu transitions
  useEffect(() => {
    if (mobileMenuOpen) {
      // Small delay to allow CSS transitions to work properly
      setTimeout(() => {
        setMenuVisible(true);
      }, 10);
    } else {
      setMenuVisible(false);
    }
  }, [mobileMenuOpen]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
    toast.success("Logged out successfully!");
    router.push("/");
    setMobileMenuOpen(false);
  };

  const handleDashboardClick = () => {
    if (!user) {
      toast.error("Please login first to access the dashboard");
      setLogin(true);
      return;
    }
    router.push("/dashboard");
    setMobileMenuOpen(false);
  };

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
    // Close mobile menu after navigation
    setMobileMenuOpen(false);
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
      <nav className="sticky top-0 w-full bg-[rgb(10,10,10)] text-white flex items-center justify-between px-4 md:px-12 h-20 border-b border-gray-700 z-50 transition duration-300 ease-in-out backdrop-blur-lg bg-opacity-70">
        <div className="flex items-center gap-2 md:gap-4">
          <button
            onClick={() => scrollToSection("hero")}
            className="text-xl md:text-2xl font-bold tracking-wide"
          >
            VisionFlow
          </button>
          {user && (
            <span className="text-gray-300 text-xs md:text-sm">
              Welcome, {user.name}
            </span>
          )}
        </div>

        {/* Middle Links - Desktop Only */}
        <div className="hidden md:flex absolute left-1/2 transform -translate-x-1/2 space-x-8 text-lg font-medium">
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

        {/* Right Side - Always Visible */}
        <div className="flex items-center">
          {/* Contact button - Desktop Only */}
          <button
            onClick={() => scrollToSection("footer")}
            className="hidden md:block hover:text-gray-400 transition mr-6"
          >
            Contact
          </button>
          
          {/* Dashboard and Login - Responsive */}
          <div className="flex space-x-2 md:space-x-6 items-center">
            <button
              onClick={handleDashboardClick}
              className="bg-blue-500 text-white px-3 py-1.5 md:px-6 md:py-2 rounded-md text-sm md:text-base hover:bg-blue-600 transition whitespace-nowrap"
            >
              Dashboard
            </button>
            
            {user ? (
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white px-3 py-1.5 md:px-6 md:py-2 rounded-md text-sm md:text-base hover:bg-red-600 transition whitespace-nowrap"
              >
                Logout
              </button>
            ) : (
              <button
                onClick={() => setLogin(true)}
                className="bg-white text-black px-3 py-1.5 md:px-6 md:py-2 rounded-md text-sm md:text-base hover:bg-gray-100 transition whitespace-nowrap"
              >
                Log In
              </button>
            )}
          </div>
          
          {/* Hamburger Menu - Mobile Only */}
          <button 
            className="md:hidden flex flex-col space-y-1.5 ml-4 z-50"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            <span className={`block w-6 h-0.5 bg-white transition-transform duration-300 ease-out ${mobileMenuOpen ? 'rotate-45 translate-y-2' : ''}`}></span>
            <span className={`block w-6 h-0.5 bg-white transition-opacity duration-300 ease-out ${mobileMenuOpen ? 'opacity-0' : ''}`}></span>
            <span className={`block w-6 h-0.5 bg-white transition-transform duration-300 ease-out ${mobileMenuOpen ? '-rotate-45 -translate-y-2' : ''}`}></span>
          </button>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      <div 
        className={`md:hidden fixed inset-0 bg-transparent z-40 pointer-events-none transition-all duration-300 ease-out ${mobileMenuOpen ? 'visible' : 'invisible'}`}
      >
        {/* Semi-transparent overlay */}
        <div 
          className={`absolute inset-0 bg-black/30 backdrop-blur-sm transition-opacity duration-300 ease-out ${menuVisible ? 'opacity-100' : 'opacity-0'}`} 
          onClick={() => setMobileMenuOpen(false)}
          style={{ pointerEvents: mobileMenuOpen ? 'auto' : 'none' }}
        ></div>
        
        {/* 3/4 height menu panel */}
        <div 
          className={`absolute top-20 left-0 right-0 h-3/4 bg-[rgba(10,10,10,0.85)] backdrop-blur-md transition-transform duration-300 ease-out ${menuVisible ? 'translate-y-0' : '-translate-y-full'}`}
          style={{ pointerEvents: mobileMenuOpen ? 'auto' : 'none' }}
        >
          <div className="flex flex-col items-center justify-center h-full">
            <div className="flex flex-col items-center space-y-8 text-xl">
              {navItems.map((item) => (
                <button
                  key={item.name}
                  onClick={() => scrollToSection(item.id)}
                  className="hover:text-gray-400 transition-colors duration-200 cursor-pointer transform hover:scale-105"
                >
                  {item.name}
                </button>
              ))}
              <button
                onClick={() => scrollToSection("footer")}
                className="hover:text-gray-400 transition-colors duration-200 transform hover:scale-105"
              >
                Contact
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {login && <Login onClose={() => setLogin(false)} />}
    </>
  );
}