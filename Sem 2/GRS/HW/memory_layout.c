#include <stdio.h>
#include <stdlib.h>

// Initialized global variable → Data segment
int global_init = 100;

// Uninitialized global variable → BSS segment
int global_uninit;

// Function to represent code segment
void sampleFunction() {
    // Just a dummy function
}

int main() {

    // Local variable → Stack
    int local_var = 10;

    // Dynamically allocated memory → Heap
    int *heap_var = (int *)malloc(sizeof(int));

    printf("===== Process Memory Layout Addresses =====\n\n");

    // Code segment
    printf("Address of sampleFunction (Code/Text segment): %p\n", sampleFunction);

    // Data segment (initialized global)
    printf("Address of global_init (Data segment): %p\n", &global_init);

    // BSS segment (uninitialized global)
    printf("Address of global_uninit (BSS segment): %p\n", &global_uninit);

    // Heap segment
    printf("Address of heap_var (Heap segment): %p\n", heap_var);

    // Stack segment
    printf("Address of local_var (Stack segment): %p\n", &local_var);

    free(heap_var);

    return 0;
}