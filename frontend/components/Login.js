'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';

export default function Login({ onClose }) {
    const router = useRouter();
    const [state, setState] = useState('Login');
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [image, setImage] = useState(null);
    const [isTextDataSubmitted, setIsTextDataSubmitted] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (state === 'Sign Up' && !isTextDataSubmitted) {
            return setIsTextDataSubmitted(true);
        }
        // Handle login/signup logic here
    };

    useEffect(() => {
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, []);

    return (
        <div
            className='fixed inset-0 font-semibold flex items-center justify-center bg-black/30 backdrop-blur-sm'
            onClick={onClose}
        >
            <form
                onClick={(e) => e.stopPropagation()}
                className='bg-[rgb(209,209,209)] p-8 rounded-lg shadow-lg w-96 text-black relative'
            >
                <button
                    type='button'
                    className='absolute top-4 right-4 text-gray-600'
                    onClick={onClose}
                >
                    ✕
                </button>
                <div className=' flex flex-col gap-2'>
                    <h1 className='text-2xl font-bold text-center'>{state}</h1>
                    <p className='text-sm font-semibold text-center mb-4'>Welcome back! Please sign in to continue</p>
                </div>
                {state === 'Sign Up' && isTextDataSubmitted ? (
                    <div className='flex flex-col items-center gap-4 mb-6'>
                        <label htmlFor='image' className='cursor-pointer'>
                            <Image
                                src={image ? URL.createObjectURL(image) : '/placeholder-image.png'}
                                alt='Upload'
                                width={64}
                                height={64}
                                className='rounded-full border'
                            />
                            <input type='file' id='image' hidden onChange={(e) => setImage(e.target.files[0])} />
                        </label>
                        <p className='text-sm'>Upload Company Logo</p>
                    </div>
                ) : (
                    <>
                        {state === 'Sign Up' && (
                            <input
                                type='text'
                                placeholder='Company Name'
                                className='w-full px-4 py-2 border rounded-lg mb-3'
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                required
                            />
                        )}
                        <input
                            type='email'
                            placeholder='Email'
                            className='w-full px-4 py-2 border rounded-lg mb-3'
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                        <input
                            type='password'
                            placeholder='Password'
                            className='w-full px-4 py-2 border rounded-lg mb-3'
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </>
                )}
                {state === 'Login' && (
                    <p className='text-sm font-semibold text-blue-600 text-right cursor-pointer mb-3'>Forgot password?</p>
                )}
                <button className='w-full bg-blue-600 text-white py-2 text-xl font-bold rounded-lg hover:bg-blue-700'>
                    {state === 'Login' ? 'Login' : isTextDataSubmitted ? 'Create Account' : 'Next'}
                </button>
                <p className='text-center font-semibold text-sm mt-4'>
                    {state === 'Login' ? (
                        <>Don't have an account? <span className='text-blue-600 cursor-pointer' onClick={() => setState('Sign Up')}>Sign Up</span></>
                    ) : (
                        <>Already have an account? <span className='text-blue-600 cursor-pointer' onClick={() => setState('Login')}>Login</span></>
                    )}
                </p>
            </form>
        </div>
    );
}