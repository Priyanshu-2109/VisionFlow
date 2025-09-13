"use client";
import React from "react";
import Squares from "./ui/Squares"; // Import the Squares component
import Link from "next/link";

export default function HeroSection() {
  return (
    <section id="hero" className="relative bg-[rgb(10,10,10)] text-white min-h-screen flex pt-20 items-center px-4 sm:px-8 md:px-12 overflow-hidden">
      {/* Squares Background */}
      <div className="absolute inset-0 z-0">
        <Squares
          speed={0.5}
          squareSize={40}
          direction="diagonal" // up, down, left, right, diagonal
          borderColor="rgba(255, 255, 255, 0.2)" // Lighter color
          // hoverFillColor="rgba(255, 255, 255, 0.7)" // Subtle hover effect
        />
        {/* Gradient Fade-Out */}
        <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-b from-transparent to-[rgb(10,10,10)] z-10"></div>
      </div>

      {/* Content */}
      <div className="relative z-20 max-w-3xl">
        <h1 className="text-5xl sm:text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
          Transform Data into Business Growth
        </h1>
        <p className="text-base sm:text-lg md:text-xl text-gray-400 mt-3 sm:mt-4 md:mt-6">
          Empower your business with our CRM tool - streamline customer
          interactions, boost sales, and drive growth effortlessly.
        </p>
        <div className="mt-6 sm:mt-8 flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4">
          <button className="bg-white text-black px-4 sm:px-6 py-2.5 sm:py-3 text-base sm:text-lg font-medium rounded-md flex items-center justify-center">
            <Link href={"/dashboard"} className="w-full text-center">
              Get Started →
            </Link>
          </button>
          <button className="border border-gray-500 text-white px-4 sm:px-6 py-2.5 sm:py-3 text-base sm:text-lg font-medium rounded-md flex items-center justify-center">
            <Link href={"/contactsales"} className="w-full text-center">
              Contact Sales →
            </Link>
          </button>
        </div>
      </div>
    </section>
  );
}
